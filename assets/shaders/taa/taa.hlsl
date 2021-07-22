#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"
#include "../inc/image.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/unjitter_taa.hlsl"
#include "../inc/soft_color_clamp.hlsl"
#include "taa_common.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> reprojection_tex;
[[vk::binding(3)]] Texture2D<float> depth_tex;
[[vk::binding(4)]] RWTexture2D<float4> output_tex;
[[vk::binding(5)]] RWTexture2D<float4> debug_output_tex;
[[vk::binding(6)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

// Apply at Mitchell-Netravali filter to the current frame, "un-jittering" it,
// and sharpening the content.
#define FILTER_CURRENT_FRAME 1
#define USE_ACCUMULATION 1
#define RESET_ACCUMULATION 0
#define USE_NEIGHBORHOOD_CLAMPING 1
#define TARGET_SAMPLE_COUNT 6


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

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    //debug_output_tex[px] = 0;

    const float2 input_resolution_scale = input_tex_size.xy / output_tex_size.xy;
    float2 uv = get_uv(px, output_tex_size);

    float4 history_packed = history_tex[px];
    float3 history = history_packed.rgb;
    float history_coverage = max(0.0, history_packed.a);

    float4 bhistory_packed = fetch_blurred_history(px, 2, 1);
    float3 bhistory = bhistory_packed.rgb;
    float3 bhistory_coverage = bhistory_packed.a;

    history = rgb_to_ycbcr(history);
    bhistory = rgb_to_ycbcr(bhistory);

    UnjitteredSampleInfo center_sample = sample_image_unjitter_taa(
        TextureImage::from_parts(input_tex, input_tex_size.xy),
        px,
        output_tex_size.xy,
        frame_constants.view_constants.sample_offset_pixels,
        UnjitterSettings::make_default(),
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

#if RESET_ACCUMULATION
    history_coverage = 0;
#endif

    float coverage = 1;
#if FILTER_CURRENT_FRAME
    float3 center = center_sample.color.rgb;
    coverage = center_sample.coverage;
#else
    float3 center = rgb_to_ycbcr(decode_rgb(input_tex[px].rgb));
#endif

    float3 bcenter = bcenter_sample.color.rgb / bcenter_sample.coverage;

    history = lerp(history, bcenter, saturate(1.0 - history_coverage));
    bhistory = lerp(bhistory, bcenter, saturate(1.0 - bhistory_coverage));

    //debug_output_tex[px] = float4(abs(history - bhistory), 1);
    //debug_output_tex[px] = float4(encode_rgb(ycbcr_to_rgb(bcenter)), 1);
    //debug_output_tex[px] = float4(encode_rgb(ycbcr_to_rgb(bhistory)), 1);
    //debug_output_tex[px] = float4(abs(center/coverage - bcenter), 1);
    //debug_output_tex[px] = float4(abs(encode_rgb(ycbcr_to_rgb(bcenter)) - encode_rgb(ycbcr_to_rgb(bhistory))), 1);

    float3 ex = center_sample.ex;
    float3 ex2 = center_sample.ex2;
	float3 dev = sqrt(max(0.0.xxx, ex2 - ex * ex));

    float local_contrast = dev.x / (ex.x + 1e-5);
    float box_size = 1.0;
    box_size *= lerp(0.5, 1.0, smoothstep(-0.1, 0.3, local_contrast));
    //box_size *= lerp(0.5, 1.0, clamp(1.0 - texel_center_dist, 0.0, 1.0));

    const float n_deviations = 1.5;// * lerp(0.75, 1.0, reproj.w) * resolution_scale_dev_boost;

	float3 nmin = ex - dev * box_size * n_deviations;
	float3 nmax = ex + dev * box_size * n_deviations;

    float blend_factor = 1.0;
    
	#if USE_ACCUMULATION
        // TODO: make better use of the quad reprojection validity
        //uint quad_reproj_valid_packed = uint(reproj.z * 15.0 + 0.5);
        //float4 quad_reproj_valid = (quad_reproj_valid_packed & uint4(1, 2, 4, 8)) != 0;
        //blend_factor = lerp(1.0, 1.0 / 12.0, dot(quad_reproj_valid, 0.25));
        //blend_factor = lerp(1.0, 1.0 / 12.0, min(1.0, dot(quad_reproj_valid, 1.0)));
        blend_factor = 1.0 / TARGET_SAMPLE_COUNT;

        // HACK: when used with quad rejection, this reduces shimmering,
        // but increases ghosting; mostly useful for upsampling
        //blend_factor = min(blend_factor, WaveReadLaneAt(blend_factor, WaveGetLaneIndex() ^ 1));
        //blend_factor = min(blend_factor, WaveReadLaneAt(blend_factor, WaveGetLaneIndex() ^ 8));

        //uint quad_reproj_valid_packed = uint(reproj.z * 15.0 + 0.5);
        //float4 quad_reproj_valid = (quad_reproj_valid_packed & uint4(1, 2, 4, 8)) != 0;
        //result = quad_reproj_valid.rgb;

    #if USE_NEIGHBORHOOD_CLAMPING
        float3 clamped_history = clamp(bhistory, nmin, nmax);
    #else
		float3 clamped_history = bhistory;
    #endif

        // "Anti-flicker"
        {
            float clamp_dist = (min(abs(history.x - nmin.x), abs(history.x - nmax.x))) / max(max(history.x, ex.x), 1e-5);
            blend_factor *= lerp(0.2, 1.0, smoothstep(0.0, 2.0, clamp_dist));
        }

        float3 outlier3 = max(0.0, (max(nmin - history, history - nmax)) / (0.1 + max(max(abs(history), abs(ex)), 1e-5)));
        float3 boutlier3 = max(0.0, (max(nmin - bhistory, bhistory - nmax)) / (0.1 + max(max(abs(bhistory), abs(ex)), 1e-5)));

        float outlier = max(outlier3.x, max(outlier3.y, outlier3.z));
        float boutlier = max(boutlier3.x, max(boutlier3.y, boutlier3.z));

        //float soutlier = saturate(coverage * outlier + boutlier - coverage * saturate(outlier) * saturate(boutlier));
        //float soutlier = saturate(boutlier);
        float soutlier = saturate(lerp(boutlier, outlier, coverage));

        const uint2 reproj_px = uint2((px + 0.5) * input_resolution_scale);
        float2 reproj_xy = reprojection_tex[reproj_px].xy;

#if 1
        {
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
            
            //history_coverage *= 1 - 0.25 * saturate(bclamp_amount);
            history_coverage *= keep_detail;

            //debug_output_tex[px] = float4(saturate(outlier), saturate(boutlier), soutlier, 1);
            //debug_output_tex[px] = float4(edge_outliers, 0, 0, 1);
            //debug_output_tex[px] = float4(outlier.xxx * 100, 1);
            //debug_output_tex[px] = float4(0, saturate(stabilize_edges).x, 0, 0.9);
            //debug_output_tex[px] = float4(bclamp_amount.xxx, 1);
            //debug_output_tex[px] = float4(bclamp_as_lerp.xxx, 1);
            //debug_output_tex[px] = float4(keep_detail.xxx, 1);
            //debug_output_tex[px] = float4(1 - (saturate(bclamp_as_lerp) * (1 - stabilize_edges)), 0, 0, 1);
            //debug_output_tex[px] = float4(history_only_edges.xxx, 1);
            //debug_output_tex[px] = float4(10 * abs(clamp_diff.xxx), 1);
            //debug_output_tex[px] = float4(ycbcr_to_rgb(clamped_history), 1);
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
#endif

        //debug_output_tex[px] = float4(ycbcr_to_rgb(bhistory), 1);
        //debug_output_tex[px] = float4(ycbcr_to_rgb(bcenter), 1);

        // Bias towards history / blurred center at the start of accumulation.
        // Blurry new pixels are better than noisy new pixels.
        history_coverage = max(history_coverage, 1);

        float total_coverage = max(1e-5, history_coverage + coverage);
		float3 result = lerp(clamped_history, center / max(1e-5, coverage), blend_factor);
        //float3 result = (clamped_history * history_coverage + center) / total_coverage;

        const float max_coverage = TARGET_SAMPLE_COUNT;
        total_coverage = min(max_coverage, total_coverage);

        result = ycbcr_to_rgb(result);
		result = encode_rgb(result);
        coverage = total_coverage;
	#else
		float3 result = encode_rgb(ycbcr_to_rgb(center / coverage));
	#endif

    output_tex[px] = float4(result, coverage);
}
