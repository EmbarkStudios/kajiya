#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"
#include "../inc/image.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/unjitter_taa.hlsl"
#include "taa_common.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> reprojection_tex;
[[vk::binding(3)]] Texture2D<float2> closest_velocity_tex;
[[vk::binding(4)]] Texture2D<float2> velocity_history_tex;
[[vk::binding(5)]] Texture2D<float> depth_tex;
[[vk::binding(6)]] Texture2D<float3> smooth_var_history_tex;
[[vk::binding(7)]] Texture2D<float> input_prob_tex;
[[vk::binding(8)]] RWTexture2D<float4> temporal_output_tex;
[[vk::binding(9)]] RWTexture2D<float4> output_tex;
[[vk::binding(10)]] RWTexture2D<float3> smooth_var_output_tex;
[[vk::binding(11)]] RWTexture2D<float2> velocity_output_tex;
[[vk::binding(12)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

// Apply at spatial kernel to the current frame, "un-jittering" it.
#define FILTER_CURRENT_FRAME 1

#define USE_ACCUMULATION 1
#define RESET_ACCUMULATION 0
#define USE_NEIGHBORHOOD_CLAMPING 1
#define TARGET_SAMPLE_COUNT 8

// If 1, outputs the input verbatim
// if N > 1, exponentially blends approximately N frames together without any clamping
#define SHORT_CIRCUIT 0

// Whether to use the input probability calculated in `input_prob.hlsl` and the subsequent filters.
// Necessary for stability of temporal super-resolution.
#define USE_CONFIDENCE_BASED_HISTORY_BLEND 1

#define INPUT_TEX input_tex
#define INPUT_REMAP InputRemap

// Draw a rectangle indicating the current frame index. Useful for debugging frame drops.
#define USE_FRAME_INDEX_INDICATOR_BAR 0


struct InputRemap {
    static InputRemap create() {
        InputRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return float4(rgb_to_ycbcr(decode_rgb(v.rgb)), 1);
    }
};

float4 fetch_blurred_history(int2 px, int k, float sigma) {
    const float3 center = history_tex[px].rgb;

    float4 csum = 0;
    float wsum = 0;

    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 c = history_tex[px + int2(x, y)];
            float2 offset = float2(x, y) * sigma;
            float w = exp(-dot(offset, offset));
            float color_diff =
                linear_to_perceptual(calculate_luma(c.rgb))
                - linear_to_perceptual(calculate_luma(center));
            csum += c * w;
            wsum += w;
        }
    }

    return csum / wsum;
}

struct HistoryRemap {
    static HistoryRemap create() {
        HistoryRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return float4(rgb_to_ycbcr(v.rgb), 1);
    }
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    #if USE_FRAME_INDEX_INDICATOR_BAR
        if (px.y < 50) {
            float4 val = 0;
            if (px.x < frame_constants.frame_index * 10 % uint(output_tex_size.x)) {
                val = 1;
            }
            temporal_output_tex[px] = val;
            output_tex[px] = val;
            return;
        }
    #endif

    const float2 input_resolution_fraction = input_tex_size.xy / output_tex_size.xy;
    const uint2 reproj_px = uint2((px + 0.5) * input_resolution_fraction);
    //const uint2 reproj_px = uint2(px * input_resolution_fraction + 0.5);

    #if SHORT_CIRCUIT
        temporal_output_tex[px] = lerp(input_tex[reproj_px], float4(encode_rgb(history_tex[px].rgb), 1), 1.0 - 1.0 / SHORT_CIRCUIT);
        output_tex[px] = temporal_output_tex[px];
        return;
    #endif

    float2 uv = get_uv(px, output_tex_size);

    float4 history_packed = history_tex[px];
    float3 history = history_packed.rgb;
    float history_coverage = max(0.0, history_packed.a);

    float4 bhistory_packed = fetch_blurred_history(px, 2, 1);
    float3 bhistory = bhistory_packed.rgb;
    float3 bhistory_coverage = bhistory_packed.a;

    history = rgb_to_ycbcr(history);
    bhistory = rgb_to_ycbcr(bhistory);

    const float4 reproj = reprojection_tex[reproj_px];
    const float2 reproj_xy = closest_velocity_tex[px];

    UnjitteredSampleInfo center_sample = sample_image_unjitter_taa(
        TextureImage::from_parts(INPUT_TEX, input_tex_size.xy),
        px,
        output_tex_size.xy,
        frame_constants.view_constants.sample_offset_pixels,
        UnjitterSettings::make_default().with_kernel_half_width_pixels(1),
        INPUT_REMAP::create()
    );

    UnjitteredSampleInfo bcenter_sample = sample_image_unjitter_taa(
        TextureImage::from_parts(INPUT_TEX, input_tex_size.xy),
        px,
        output_tex_size.xy,
        frame_constants.view_constants.sample_offset_pixels,
        UnjitterSettings::make_default().with_kernel_scale(0.333),
        INPUT_REMAP::create()
    );

    float coverage = 1;
#if FILTER_CURRENT_FRAME
    float3 center = center_sample.color.rgb;
    coverage = center_sample.coverage;
#else
    float3 center = rgb_to_ycbcr(decode_rgb(INPUT_TEX[px].rgb));
#endif

    float3 bcenter = bcenter_sample.color.rgb / bcenter_sample.coverage;

    history = lerp(history, bcenter, saturate(1.0 - history_coverage));
    bhistory = lerp(bhistory, bcenter, saturate(1.0 - bhistory_coverage));

    const float input_prob = input_prob_tex[reproj_px];

    float3 ex = center_sample.ex;
    float3 ex2 = center_sample.ex2;
    const float3 var = max(0.0.xxx, ex2 - ex * ex);

    const float3 prev_var = smooth_var_history_tex.SampleLevel(sampler_lnc, uv + reproj_xy, 0).x;

    // TODO: factor-out camera-only velocity
    const float2 vel_now = closest_velocity_tex[px] / frame_constants.delta_time_seconds;
    const float2 vel_prev = velocity_history_tex.SampleLevel(sampler_llc, uv + closest_velocity_tex[px], 0);
    const float vel_diff = length((vel_now - vel_prev) / max(1, abs(vel_now + vel_prev)));
    const float var_blend = saturate(0.3 + 0.7 * (1 - reproj.z) + vel_diff);

    float3 smooth_var = max(var, lerp(prev_var, var, var_blend));

    const float var_prob_blend = saturate(input_prob);
    smooth_var = lerp(var, smooth_var, var_prob_blend);
    
    const float3 input_dev = sqrt(var);

    float4 this_frame_result = 0;
    #define DEBUG_SHOW(value) { this_frame_result = float4((float3)(value), 1); }

    float3 clamped_history;

    // Perform neighborhood clamping / disocclusion rejection
    {
        // Use a narrow color bounding box to avoid disocclusions
        float box_n_deviations = 0.8;

        if (USE_CONFIDENCE_BASED_HISTORY_BLEND) {
            // Expand the box based on input confidence.
            box_n_deviations = lerp(box_n_deviations, 3, input_prob);
        }

    	float3 nmin = ex - input_dev * box_n_deviations;
    	float3 nmax = ex + input_dev * box_n_deviations;

    	#if USE_ACCUMULATION
        #if USE_NEIGHBORHOOD_CLAMPING
            float3 clamped_bhistory = clamp(bhistory, nmin, nmax);
        #else
    		float3 clamped_bhistory = bhistory;
        #endif

            const float clamping_event = length(max(0.0, max(bhistory - nmax, nmin - bhistory)) / max(0.01, ex));

            float3 outlier3 = max(0.0, (max(nmin - history, history - nmax)) / (0.1 + max(max(abs(history), abs(ex)), 1e-5)));
            float3 boutlier3 = max(0.0, (max(nmin - bhistory, bhistory - nmax)) / (0.1 + max(max(abs(bhistory), abs(ex)), 1e-5)));

            // Temporal outliers in sharp history
            float outlier = max(outlier3.x, max(outlier3.y, outlier3.z));
            //DEBUG_SHOW(outlier);

            // Temporal outliers in blurry history
            float boutlier = max(boutlier3.x, max(boutlier3.y, boutlier3.z));
            //DEBUG_SHOW(boutlier);

            const bool history_valid = all(uv + reproj_xy == saturate(uv + reproj_xy));

    #if 1
            if (history_valid) {
                const float non_disoccluding_outliers = max(0.0, outlier - boutlier) * 10;
                //DEBUG_SHOW(non_disoccluding_outliers);

                const float3 unclamped_history_detail = history - clamped_bhistory;

                // Temporal luminance diff, containing history edges, and peaking when
                // clamping happens.
                const float temporal_clamping_detail = length(unclamped_history_detail.x / max(1e-3, input_dev.x)) * 0.05;
                //DEBUG_SHOW(temporal_clamping_detail);

                // Close to 1.0 when temporal clamping is relatively low. Close to 0.0 when disocclusions happen.
                const float temporal_stability = saturate(1 - temporal_clamping_detail);
                //DEBUG_SHOW(temporal_stability);

                const float allow_unclamped_detail = saturate(non_disoccluding_outliers) * temporal_stability;
                //const float allow_unclamped_detail = saturate(non_disoccluding_outliers * exp2(-length(input_tex_size.xy * reproj_xy))) * temporal_stability;
                //DEBUG_SHOW(allow_unclamped_detail);

                // Clamping happens to blurry history because input is at lower fidelity (and potentially lower resolution)
                // than history (we don't have enough data to perform good clamping of high frequencies).
                // In order to keep high-resolution detail in the output, the high-frequency content is split from
                // low-frequency (`bhistory`), and then selectively re-added. The detail needs to be attenuated
                // in order not to cause false detail (which look like excessive sharpening artifacts).
                float3 history_detail = history - bhistory;

                // Selectively stabilize some detail, allowing unclamped history
                history_detail = lerp(history_detail, unclamped_history_detail, allow_unclamped_detail);

                // 0..1 value of how much clamping initially happened in the blurry history
                const float initial_bclamp_amount = saturate(dot(
                    clamped_bhistory - bhistory, bcenter - bhistory)
                    / max(1e-5, length(clamped_bhistory - bhistory) * length(bcenter - bhistory)));

                // Ditto, after adjusting for `allow_unclamped_detail`
                const float effective_clamp_amount = saturate(initial_bclamp_amount) * (1 - allow_unclamped_detail);
                //DEBUG_SHOW(effective_clamp_amount);

                // Where clamping happened to the blurry history, also remove the detail (history-bhistory)
                const float keep_detail = 1 - effective_clamp_amount;
                history_detail *= keep_detail;

                // Finally, construct the full-frequency output.
                clamped_history = clamped_bhistory + history_detail;
                
                #if 1
                    // When temporally upsampling, after a clamping event, there's pixellation
                    // because we haven't accumulated enough samples yet from
                    // the reduced-resolution input. Dampening history coverage when
                    // clamping happens allows us to boost this convergence.

                    history_coverage *= lerp(
                        lerp(0.0, 0.9, keep_detail), 1.0, saturate(10 * clamping_event)
                    );
                #endif
            } else {
                clamped_history = clamped_bhistory;
                coverage = 1;
                center = bcenter;
                history_coverage = 0;
            }
    #else
            clamped_history = clamp(history, nmin, nmax);
    #endif

        if (USE_CONFIDENCE_BASED_HISTORY_BLEND) {
            // If input confidence is high, blend in unclamped history.
            clamped_history = lerp(
                clamped_history,
                history,
                smoothstep(0.5, 1.0, input_prob)
            );
        }
    }


    #if RESET_ACCUMULATION
        history_coverage = 0;
    #endif

        float total_coverage = max(1e-5, history_coverage + coverage);
        float3 temporal_result = (clamped_history * history_coverage + center) / total_coverage;

        const float max_coverage = max(2, TARGET_SAMPLE_COUNT / (input_resolution_fraction.x * input_resolution_fraction.y));

        total_coverage = min(max_coverage, total_coverage);

        coverage = total_coverage;
	#else
		float3 temporal_result = center / coverage;
	#endif

    smooth_var_output_tex[px] = smooth_var;

    temporal_result = ycbcr_to_rgb(temporal_result);
	temporal_result = encode_rgb(temporal_result);
    temporal_result = max(0.0, temporal_result);
    
    this_frame_result.rgb = lerp(temporal_result, this_frame_result.rgb, this_frame_result.a);

    temporal_output_tex[px] = float4(temporal_result, coverage);
    output_tex[px] = this_frame_result;

    float2 vel_out = reproj_xy;
    float vel_out_depth = 0;

    // It's critical that this uses the closest depth since it's compared to closest depth
    velocity_output_tex[px] = closest_velocity_tex[px] / frame_constants.delta_time_seconds;
}
