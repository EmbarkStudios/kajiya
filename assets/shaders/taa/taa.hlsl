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
[[vk::binding(6)]] Texture2D<float> smooth_var_history_tex;
[[vk::binding(7)]] Texture2D<float4> input_stats_tex;
[[vk::binding(8)]] Texture2D<float4> filtered_input_tex;
[[vk::binding(9)]] RWTexture2D<float4> output_tex;
[[vk::binding(10)]] RWTexture2D<float4> debug_output_tex;
[[vk::binding(11)]] RWTexture2D<float> smooth_var_output_tex;
[[vk::binding(12)]] RWTexture2D<float2> velocity_output_tex;
[[vk::binding(13)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

// Apply at Mitchell-Netravali filter to the current frame, "un-jittering" it,
// and sharpening the content.
#define FILTER_CURRENT_FRAME 1
#define USE_ACCUMULATION 1
#define RESET_ACCUMULATION 0
#define USE_NEIGHBORHOOD_CLAMPING 1
#define USE_ANTIFLICKER 0
#define TARGET_SAMPLE_COUNT 8
#define SHORT_CIRCUIT 0
#define USE_FILTERED_INPUT 0
#define USE_CONFIDENCE_BASED_HISTORY_BLEND 1

#if USE_FILTERED_INPUT
    #define INPUT_TEX filtered_input_tex
    #define INPUT_REMAP IdentityImageRemap
#else
    #define INPUT_TEX input_tex
    #define INPUT_REMAP InputRemap
#endif

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
            output_tex[px] = val;
            debug_output_tex[px] = val;
            return;
        }
    #endif

    const float2 input_resolution_fraction = input_tex_size.xy / output_tex_size.xy;
    const uint2 reproj_px = uint2((px + 0.5) * input_resolution_fraction);
    //const uint2 reproj_px = uint2(px * input_resolution_fraction + 0.5);

    #if SHORT_CIRCUIT
        output_tex[px] = lerp(filtered_input_tex[reproj_px], float4(encode_rgb(history_tex[px].rgb), 1), 1.0 - 1.0 / SHORT_CIRCUIT);
        debug_output_tex[px] = output_tex[px];
        return;
    #endif

    float3 debug_out = 0;

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

    const float4 input_stats = input_stats_tex[reproj_px];
    const float filtered_input_dev = input_stats.y;
    const float input_prob = input_stats.x;

    float3 ex = center_sample.ex;
    float3 ex2 = center_sample.ex2;
    const float3 var = max(0.0.xxx, ex2 - ex * ex);

    const float prev_meta = smooth_var_history_tex.SampleLevel(sampler_lnc, uv + reproj_xy, 0);
    float prev_var = prev_meta.x;

    const float2 vel_now = closest_velocity_tex[px] / frame_constants.delta_time_seconds;
    const float2 vel_prev = velocity_history_tex.SampleLevel(sampler_llc, uv + closest_velocity_tex[px], 0);
    const float vel_diff = length((vel_now - vel_prev) / max(1, abs(vel_now + vel_prev)));
    const float var_blend = saturate(0.3 + 0.7 * (1 - reproj.z) + vel_diff);

    float smooth_var = max(var.x, lerp(prev_var, var.x, var_blend));

    const float var_prob_blend = saturate(input_prob);
    smooth_var = lerp(var.x, smooth_var, var_prob_blend);
    
    const float3 input_dev = sqrt(var);

    //float local_contrast = input_dev.x / (ex.x + 1e-5);
    float box_n_deviations = 0.8;
    //box_n_deviations *= lerp(0.5, 1.0, smoothstep(-0.1, 0.3, local_contrast));
    //box_n_deviations *= lerp(0.5, 1.0, clamp(1.0 - texel_center_dist, 0.0, 1.0));

	float3 nmin = ex - input_dev * box_n_deviations;
	float3 nmax = ex + input_dev * box_n_deviations;

	#if USE_ACCUMULATION
    #if USE_NEIGHBORHOOD_CLAMPING
        float3 clamped_history = clamp(bhistory, nmin, nmax);
    #else
		float3 clamped_history = bhistory;
    #endif

        const float clamping_event = length(max(0.0, max(bhistory - nmax, nmin - bhistory)) / max(0.01, ex));

        float3 outlier3 = max(0.0, (max(nmin - history, history - nmax)) / (0.1 + max(max(abs(history), abs(ex)), 1e-5)));
        float3 boutlier3 = max(0.0, (max(nmin - bhistory, bhistory - nmax)) / (0.1 + max(max(abs(bhistory), abs(ex)), 1e-5)));

        float outlier = max(outlier3.x, max(outlier3.y, outlier3.z));
        float boutlier = max(boutlier3.x, max(boutlier3.y, boutlier3.z));
        float soutlier = saturate(lerp(boutlier, outlier, coverage));

        const bool history_valid = all(uv + reproj_xy == saturate(uv + reproj_xy));

#if 1
        if (history_valid) {
            const float bclamp_amount = length((clamped_history - bhistory) / max(1e-5, abs(ex)));
            const float edge_outliers = abs(boutlier - outlier) * 10;
            const float non_edge_outliers = (boutlier - abs(boutlier - outlier)) * 10;

            const float bclamp_as_lerp = dot(
                clamped_history - bhistory, bcenter - bhistory)
                / max(1e-5, length(clamped_history - bhistory) * length(bcenter - bhistory));

            float3 clamp_diff = history - clamped_history;
            float3 diff = history - bhistory;

            const float history_only_edges = length(clamp_diff.x / max(1e-3, input_dev.x)) * 0.05;
            const float stabilize_edges = saturate(edge_outliers * exp2(-length(input_tex_size.xy * reproj_xy))) * saturate(1 - history_only_edges);
            diff = lerp(diff, clamp_diff, stabilize_edges);

            diff.yz /= max(1e-5, abs(diff.x));

            const float keep_detail = 1 - saturate(bclamp_as_lerp) * (1 - stabilize_edges);
            diff.x *= keep_detail;
            clamped_history = clamped_history + diff * float3(1.0, max(1e-5, abs(diff.xx)));
            
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
            coverage = 1;
            center = bcenter;
            history_coverage = 0;
        }
#else
        clamped_history = clamp(history, nmin, nmax);
        //clamped_history = history;
#endif

    #if USE_CONFIDENCE_BASED_HISTORY_BLEND
        clamped_history = lerp(
            clamped_history,
            history,
            smoothstep(0.2, 1.0, input_prob)
        );
    #endif


    #if RESET_ACCUMULATION
        history_coverage = 0;
    #endif

        float total_coverage = max(1e-5, history_coverage + coverage);
        float3 result = (clamped_history * history_coverage + center) / total_coverage;

        const float max_coverage = max(2, TARGET_SAMPLE_COUNT / (input_resolution_fraction.x * input_resolution_fraction.y));

        total_coverage = min(max_coverage, total_coverage);

        coverage = total_coverage;
	#else
		float3 result = center / coverage;
	#endif

    // "Anti-flicker"
    if (USE_ANTIFLICKER) {
        float clamp_dist = (min(abs(bhistory.x - nmin.x), abs(bhistory.x - nmax.x))) / max(max(bhistory.x, ex.x), 1e-5);
        const float blend_mult = lerp(0.2, 1.0, smoothstep(0.0, 2.0, clamp_dist));
        result = lerp(clamped_history, result, blend_mult);
    }

    smooth_var_output_tex[px] = smooth_var;

    result = ycbcr_to_rgb(result);
	result = encode_rgb(result);
    
    debug_out = result;

    output_tex[px] = float4(max(0.0, result), coverage);
    debug_output_tex[px] = float4(debug_out, 1);

    float2 vel_out = reproj_xy;
    float vel_out_depth = 0;

    // It's critical that this uses the closest depth since it's compared to closest depth
    velocity_output_tex[px] = closest_velocity_tex[px] / frame_constants.delta_time_seconds;
}
