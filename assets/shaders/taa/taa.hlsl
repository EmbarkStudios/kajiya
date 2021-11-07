#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"
#include "../inc/image.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/unjitter_taa.hlsl"
#include "../inc/soft_color_clamp.hlsl"
#include "taa_common.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> reprojection_tex;
[[vk::binding(3)]] Texture2D<float2> closest_velocity_tex;
[[vk::binding(4)]] Texture2D<float2> velocity_history_tex;
[[vk::binding(5)]] Texture2D<float> depth_tex;
[[vk::binding(6)]] Texture2D<float4> meta_history_tex;
[[vk::binding(7)]] Texture2D<float4> input_stats_tex;
[[vk::binding(8)]] RWTexture2D<float4> output_tex;
[[vk::binding(9)]] RWTexture2D<float4> debug_output_tex;
[[vk::binding(10)]] RWTexture2D<float4> meta_output_tex;
[[vk::binding(11)]] RWTexture2D<float2> velocity_output_tex;
[[vk::binding(12)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

// Apply at Mitchell-Netravali filter to the current frame, "un-jittering" it,
// and sharpening the content.
#define FILTER_CURRENT_FRAME 1
#define USE_ACCUMULATION 1
#define RESET_ACCUMULATION 0
#define USE_NEIGHBORHOOD_CLAMPING 1
#define TARGET_SAMPLE_COUNT 4
#define SHORT_CIRCUIT 1


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
            //w *= exp(-color_diff * color_diff * 100);
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
    const float2 input_resolution_scale = input_tex_size.xy / output_tex_size.xy;
    const uint2 reproj_px = uint2((px + 0.5) * input_resolution_scale);

    #if SHORT_CIRCUIT
        output_tex[px] = lerp(input_tex[reproj_px], float4(encode_rgb(history_tex[px].rgb), 1), 1.0 - 1.0 / SHORT_CIRCUIT);
        debug_output_tex[px] = output_tex[px];
        //output_tex[px] = reprojection_tex[reproj_px].zzzz * 0.1;
        return;
    #endif

    float3 debug_out = 0;

    //debug_output_tex[px] = 0;

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
    //const float2 reproj_xy = reproj.xy;
    const float2 reproj_xy = closest_velocity_tex[px];

    UnjitteredSampleInfo center_sample = sample_image_unjitter_taa(
        TextureImage::from_parts(input_tex, input_tex_size.xy),
        px,
        output_tex_size.xy,
        frame_constants.view_constants.sample_offset_pixels,
        UnjitterSettings::make_default().with_kernel_half_width_pixels(1),
        InputRemap::create()
    );

    UnjitteredSampleInfo bcenter_sample = sample_image_unjitter_taa(
        TextureImage::from_parts(input_tex, input_tex_size.xy),
        px,
        output_tex_size.xy,
        frame_constants.view_constants.sample_offset_pixels,
        UnjitterSettings::make_default().with_kernel_scale(0.333),
        InputRemap::create()
    );

    float coverage = 1;
#if FILTER_CURRENT_FRAME
    float3 center = center_sample.color.rgb;
    coverage = center_sample.coverage;
#else
    float3 center = rgb_to_ycbcr(decode_rgb(input_tex[px].rgb));
#endif

    const float3 bcenter = bcenter_sample.color.rgb / bcenter_sample.coverage;

    history = lerp(history, bcenter, saturate(1.0 - history_coverage));
    bhistory = lerp(bhistory, bcenter, saturate(1.0 - bhistory_coverage));

    //debug_output_tex[px] = float4(abs(history - bhistory), 1);
    //debug_output_tex[px] = float4(encode_rgb(ycbcr_to_rgb(bcenter)), 1);
    //debug_output_tex[px] = float4(encode_rgb(ycbcr_to_rgb(bhistory)), 1);
    //debug_output_tex[px] = float4(abs(center/coverage - bcenter), 1);
    //debug_output_tex[px] = float4(abs(encode_rgb(ycbcr_to_rgb(bcenter)) - encode_rgb(ycbcr_to_rgb(bhistory))), 1);

    float3 ex = center_sample.ex;
    float3 ex2 = center_sample.ex2;
    const float3 var = max(0.0.xxx, ex2 - ex * ex);

    const float4 prev_meta = meta_history_tex.SampleLevel(sampler_lnc, uv + reproj_xy, 0);
    float prev_var = prev_meta.x;
    float prev_ex = prev_meta.z;
    float prev_ex2 = prev_meta.w;

    const float2 vel_now = closest_velocity_tex[px] / frame_constants.delta_time_seconds;
    const float2 vel_prev = velocity_history_tex.SampleLevel(sampler_llc, uv + closest_velocity_tex[px], 0);
    float vel_diff = length((vel_now - vel_prev) / max(1, abs(vel_now + vel_prev))) > 0.2;
    /*vel_diff = max(vel_diff, WaveReadLaneAt(vel_diff, WaveGetLaneIndex() ^ 1));
    vel_diff = max(vel_diff, WaveReadLaneAt(vel_diff, WaveGetLaneIndex() ^ 2));
    vel_diff = max(vel_diff, WaveReadLaneAt(vel_diff, WaveGetLaneIndex() ^ 8));
    vel_diff = max(vel_diff, WaveReadLaneAt(vel_diff, WaveGetLaneIndex() ^ 16));
    vel_diff = max(vel_diff, WaveReadLaneAt(vel_diff, WaveGetLaneIndex() ^ 32));*/

    const float var_blend = saturate(0.3 + 0.7 * (1 - reproj.z) + vel_diff);
    //const float var_blend = 0.3;
    //const float var_blend = saturate(0.3 + 0.7 * smoothstep(3, 0, history_coverage) + 0.7 * (1 - reproj.z));

    float smooth_var = max(var.x, lerp(prev_var, var.x, var_blend));
    float smooth_ex = max(ex.x, lerp(prev_ex, ex.x, var_blend));
    float smooth_ex2 = max(ex2.x, lerp(prev_ex2, ex2.x, var_blend));

    const float input_prob = input_stats_tex[reproj_px].x;

    smooth_var = max(0.0, smooth_ex2 - smooth_ex * smooth_ex);

    const float var_prob_blend = pow(saturate(input_prob), 32);
    smooth_var = lerp(var.x, smooth_var, var_prob_blend);
    smooth_ex = lerp(ex.x, smooth_ex, var_prob_blend);
    smooth_ex2 = lerp(ex2.x, smooth_ex2, var_prob_blend);
    
    /*if (input_prob < 0.9) {
        dev = sqrt(var);
        smooth_var = var.x;
    }*/

    #if 1
        float3 dev = sqrt(var * smooth_var / max(1e-20, var.x));
    #else
        float3 dev = sqrt(var);
    #endif

    float local_contrast = dev.x / (ex.x + 1e-5);
    float box_size = 1.0;
    //box_size *= lerp(0.5, 1.0, smoothstep(-0.1, 0.3, local_contrast));
    //box_size *= lerp(0.5, 1.0, clamp(1.0 - texel_center_dist, 0.0, 1.0));

    const float n_deviations = lerp(4.0, 1.5, sqrt(input_resolution_scale.x));

	float3 nmin = ex - dev * box_size * n_deviations;
	float3 nmax = ex + dev * box_size * n_deviations;

	#if USE_ACCUMULATION
    #if USE_NEIGHBORHOOD_CLAMPING
        float3 clamped_history = clamp(bhistory, nmin, nmax);
    #else
		float3 clamped_history = bhistory;
    #endif

        const float clamping_event = length(max(0.0, max(bhistory - nmax, nmin - bhistory)) / max(0.01, ex));
        const float prev_clamping_event = prev_meta.y;
        const float smooth_clamping_event = max(clamping_event, lerp(prev_clamping_event, clamping_event, 0.1));

        float3 outlier3 = max(0.0, (max(nmin - history, history - nmax)) / (0.1 + max(max(abs(history), abs(ex)), 1e-5)));
        float3 boutlier3 = max(0.0, (max(nmin - bhistory, bhistory - nmax)) / (0.1 + max(max(abs(bhistory), abs(ex)), 1e-5)));

        float outlier = max(outlier3.x, max(outlier3.y, outlier3.z));
        float boutlier = max(boutlier3.x, max(boutlier3.y, boutlier3.z));

        //float soutlier = saturate(coverage * outlier + boutlier - coverage * saturate(outlier) * saturate(boutlier));
        //float soutlier = saturate(boutlier);
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

            const float history_only_edges = length(clamp_diff.x / max(1e-3, dev.x)) * 0.05;
            const float stabilize_edges = saturate(edge_outliers * exp2(-length(input_tex_size.xy * reproj_xy))) * saturate(1 - history_only_edges);
            diff = lerp(diff, clamp_diff, stabilize_edges);

            diff.yz /= max(1e-5, abs(diff.x));

            const float keep_detail = 1 - saturate(bclamp_as_lerp) * (1 - stabilize_edges);
            diff.x *= keep_detail;
            clamped_history = clamped_history + diff * float3(1.0, max(1e-5, abs(diff.xx)));
            //history_coverage *= lerp(0.5, 1.0, keep_detail);
            history_coverage *= lerp(
                lerp(0.0, 0.9, keep_detail), 1.0, saturate(10 * smooth_clamping_event)
            );

            //smooth_var *= saturate(1 - 0.5 * bclamp_amount);
            //smooth_var *= saturate(1 - 4 * history_only_edges);

            //debug_out = float3(saturate(outlier), saturate(boutlier), soutlier);
            //debug_out = float3(edge_outliers, 0, 0);
            //debug_out = float3(outlier.xxx * 10);
            //debug_out = float3(0, saturate(stabilize_edges).x, 0, );
            //debug_out = float3(bclamp_amount.xxx);
            //debug_out = float3(bclamp_as_lerp.xxx);
            //debug_out = float3(keep_detail.xxx);
            //debug_out = float3(1 - (saturate(bclamp_as_lerp) * (1 - stabilize_edges)), 0, 0);
            //debug_out = float3(history_only_edges.xxx);
            //debug_out = float3(10 * abs(clamp_diff.xxx));
            //debug_out = float3(ycbcr_to_rgb(clamped_history));
        } else {
            coverage = 1;
            center = bcenter;
            history_coverage = 0;
        }
#elif 0
        if (soutlier > 0.0) {
            float3 diff = history - bhistory;
            diff *= saturate(1.0 - soutlier * 0.8);
            clamped_history = lerp(bhistory, ex, soutlier) + diff;
            //clamped_history = lerp(history, ex, soutlier);
            history_coverage *= saturate(soutlier + 0.75);
            //debug_output_tex[px] = float4(saturate(outlier), saturate(boutlier), soutlier, 1);
        } else {
            //debug_output_tex[px] = float4(0, 0, 0, 1);
        }
#else
        clamped_history = history;
#endif

    #if RESET_ACCUMULATION
        history_coverage = 0;
    #endif

        float total_coverage = max(1e-5, history_coverage + coverage);
		//float3 result = lerp(clamped_history, center / max(1e-5, coverage), blend_factor);
        float3 result = (clamped_history * history_coverage + center) / total_coverage;

        //const float max_coverage = lerp(2 * TARGET_SAMPLE_COUNT / 3, TARGET_SAMPLE_COUNT, smooth_clamping_event);
        const float max_coverage = TARGET_SAMPLE_COUNT;

        total_coverage = min(max_coverage, total_coverage);

        coverage = total_coverage;
	#else
		float3 result = center / coverage;
	#endif

    //float4 meta_out = float4(smooth_var, trend_accum, bcenter.x, smooth_ex2);
    float4 meta_out = float4(smooth_var, smooth_clamping_event, smooth_ex, smooth_ex2);
    meta_output_tex[px] = meta_out;

    result = ycbcr_to_rgb(result);
	result = encode_rgb(result);

    {
        debug_out = result;
        //debug_out = encode_rgb(ycbcr_to_rgb(history));
        //debug_out = calculate_luma(debug_out);
        //debug_out = encode_rgb(ycbcr_to_rgb(history));
        //debug_out = sqrt(smooth_var.x);
        //debug_out = history_coverage / TARGET_SAMPLE_COUNT;
        //debug_out.r = saturate(10 * smoothstep(0.2, 1.0, abs(trend_accum)));
        //float dev_dist = max(0.0, max(history - nmax, nmin - history).x) / max(1e-2, dev.x) * 0.1;
        //debug_out.r += dev_dist;
        //debug_out.r = abs(trend_accum);
        //debug_out = 20 * smooth_clamping_event;

        //debug_out *= 0.1;
        //debug_out.g += smoothstep(0.0, 0.1, (1 - input_prob));
        //debug_out.g += 1 - var_prob_blend;
        //debug_out.g += vel_diff * 0.1;
        //debug_out = saturate(10 * smooth_clamping_event);
    }

    //result = float3(abs(reprojection_tex[reproj_px].xy) * 100, 0);
    //result = reproj.z != 0;
    output_tex[px] = float4(result, coverage);
    debug_output_tex[px] = float4(debug_out, 1);

    float2 vel_out = reproj_xy;
    float vel_out_depth = 0;

    // It's critical that this uses the closest depth since it's compared to closest depth
    velocity_output_tex[px] = closest_velocity_tex[px] / frame_constants.delta_time_seconds;
}
