#include "../inc/samplers.hlsl"
#include "../inc/color.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/soft_color_clamp.hlsl"

#include "../inc/working_color_space.hlsl"

#define USE_BBOX_CLAMP 1

#if 0
    // Linear accumulation, for comparisons with path tracing
    float4 pass_through(float4 v) { return v; }
    #define linear_to_working pass_through
    #define working_to_linear pass_through
    float working_luma(float3 v) { return sRGB_to_luminance(v); }
#else
    #define linear_to_working linear_rgb_to_crunched_luma_chroma
    #define working_to_linear crunched_luma_chroma_to_linear_rgb
    float working_luma(float3 v) { return v.x; }
#endif

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float2> variance_history_tex;
[[vk::binding(3)]] Texture2D<float4> reprojection_tex;
[[vk::binding(4)]] Texture2D<float2> rt_history_invalidity_tex;
[[vk::binding(5)]] RWTexture2D<float4> output_tex;
[[vk::binding(6)]] RWTexture2D<float4> history_output_tex;
[[vk::binding(7)]] RWTexture2D<float2> variance_history_output_tex;
[[vk::binding(8)]] cbuffer _ {
    float4 output_tex_size;
    float4 gbuffer_tex_size;
};


#define USE_TEMPORAL_FILTER 1

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    #if !USE_TEMPORAL_FILTER
        output_tex[px] = max(0.0, input_tex[px]);
        history_output_tex[px] = float4(max(0.0, input_tex[px].rgb), 32);
        return;
    #endif

    float2 uv = get_uv(px, output_tex_size);
    
    float4 center = linear_to_working(input_tex[px]);
    float4 reproj = reprojection_tex[px];

    const float4 history_mult = float4((frame_constants.pre_exposure_delta).xxx, 1);
    float4 history = linear_to_working(history_tex[px] * history_mult);

    //output_tex[px] = center;
    //return;
    
#if 1
	float4 vsum = 0.0.xxxx;
	float4 vsum2 = 0.0.xxxx;
	float wsum = 0.0;
    float hist_diff = 0.0;
    float hist_vsum = 0.0;
    float hist_vsum2 = 0.0;

    //float dev_sum = 0.0;

	const int k = 2;
    {for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = linear_to_working(input_tex[px + int2(x, y)]);
            float4 hist_neigh = linear_to_working(history_tex[px + int2(x, y)] * history_mult);

            float neigh_luma = working_luma(neigh.rgb);
            float hist_luma = working_luma(hist_neigh.rgb);

			float w = exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));
			vsum += neigh * w;
			vsum2 += neigh * neigh * w;
			wsum += w;

            //dev_sum += neigh.a * neigh.a * w;

            //hist_diff += (neigh_luma - hist_luma) * (neigh_luma - hist_luma) * w;
            hist_diff += abs(neigh_luma - hist_luma) / max(1e-5, neigh_luma + hist_luma) * w;
            hist_vsum += hist_luma * w;
            hist_vsum2 += hist_luma * hist_luma * w;
        }
    }}

	float4 ex = vsum / wsum;
	float4 ex2 = vsum2 / wsum;
	float4 dev = sqrt(max(0.0.xxxx, ex2 - ex * ex));

    hist_diff /= wsum;
    hist_vsum /= wsum;
    hist_vsum2 /= wsum;
    //dev_sum /= wsum;

    const float2 moments_history =
        variance_history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0)
        * float2(frame_constants.pre_exposure_delta, frame_constants.pre_exposure_delta * frame_constants.pre_exposure_delta);
        
    //const float center_luma = working_luma(center.rgb);
    const float center_luma = working_luma(center.rgb) + (hist_vsum - working_luma(ex.rgb));// - 0.5 * working_luma(control_variate.rgb));
    const float2 current_moments = float2(center_luma, center_luma * center_luma);
    variance_history_output_tex[px] = max(0.0, lerp(moments_history, current_moments, 0.25));
    const float center_temporal_dev = sqrt(max(0.0, moments_history.y - moments_history.x * moments_history.x));

    float center_dev = center.a;

    // Spatial-only variance estimate (dev.rgb) has halos around edges (lighting discontinuities)
    
    // Temporal variance estimate with a spatial boost
    // TODO: this version reduces flicker in pica and on skeletons in battle, but has halos in cornell_box
    //dev.rgb = center_dev * dev.rgb / max(1e-8, clamp(working_luma(dev.rgb), center_dev * 0.1, center_dev * 3.0));

    // Spatiotemporal variance estimate
    // TODO: this version seems to work best, but needs to take care near sky
    // TODO: also probably needs to be rgb :P
    //dev.rgb = sqrt(dev_sum);

    // Temporal variance estimate with spatial colors
    //dev.rgb *= center_dev / max(1e-8, working_luma(dev.rgb));

    float3 hist_dev = sqrt(abs(hist_vsum2 - hist_vsum * hist_vsum));
    //dev.rgb *= 0.1 / max(1e-5, clamp(hist_dev, dev.rgb * 0.1, dev.rgb * 10.0));

    //float temporal_change = abs(hist_vsum - working_luma(ex.rgb)) / max(1e-8, hist_vsum + working_luma(ex.rgb));
    float temporal_change = abs(hist_vsum - working_luma(ex.rgb)) / max(1e-8, hist_vsum + working_luma(ex.rgb));
    //float temporal_change = 0.1 * abs(hist_vsum - working_luma(ex.rgb)) / max(1e-5, working_luma(dev.rgb));
    //temporal_change = 0.02 * temporal_change / max(1e-5, working_luma(dev.rgb));
    //temporal_change = WaveActiveSum(temporal_change) / WaveActiveSum(1);
#endif

    const float rt_invalid = saturate(sqrt(rt_history_invalidity_tex[px / 2].x) * 4);
    const float current_sample_count = history.a;

    float clamp_box_size = 1
        * lerp(0.25, 2.0, 1.0 - rt_invalid)
        * lerp(0.333, 1.0, saturate(reproj.w))
        * 2
        ;
    clamp_box_size = max(clamp_box_size, 0.5);

	float4 nmin = center - dev * clamp_box_size;
	float4 nmax = center + dev * clamp_box_size;

#if 0
    {
    	float4 nmin2 = center;
    	float4 nmax2 = center;

    	{const int k = 2;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float4 neigh = linear_to_working(input_tex[px + int2(x, y)]);
    			nmin2 = min(nmin2, neigh);
                nmax2 = max(nmax2, neigh);
            }
        }}

        float3 nmid = lerp(nmin2.rgb, nmax2.rgb, 0.5);
        nmin2.rgb = lerp(nmid, nmin2.rgb, 1.0);
        nmax2.rgb = lerp(nmid, nmax2.rgb, 1.0);

        nmin = max(nmin, nmin2);
        nmax = min(nmax, nmax2);
    }
#endif

#if 1
	float4 clamped_history = float4(clamp(history.rgb, nmin.rgb, nmax.rgb), history.a);
#else
    float4 clamped_history = float4(
        soft_color_clamp(center.rgb, history.rgb, ex.rgb, clamp_box_size * dev.rgb),
        history.a
    );
#endif

    /*const float3 history_dist = abs(history.rgb - ex.rgb) / max(0.1, dev.rgb * 0.5);
    const float3 closest_pt = clamp(history.rgb, center.rgb - dev.rgb * 0.5, center.rgb + dev.rgb * 0.5);
    clamped_history = float4(
        lerp(history.rgb, closest_pt, lerp(0.1, 1.0, smoothstep(1.0, 3.0, history_dist))),
        history.a
    );*/

#if !USE_BBOX_CLAMP
    clamped_history = history;
#endif

    const float variance_adjusted_temporal_change = smoothstep(0.1, 1.0, 0.05 * temporal_change / center_temporal_dev);

    float max_sample_count = 32;
    max_sample_count = lerp(max_sample_count, 4, variance_adjusted_temporal_change);
    //max_sample_count = lerp(max_sample_count, 1, smoothstep(0.01, 0.6, 10 * temporal_change * (center_dev / max(1e-5, center_luma))));
    max_sample_count *= lerp(1.0, 0.5, rt_invalid);

// hax
//max_sample_count = 32;

    float3 res = lerp(clamped_history.rgb, center.rgb, 1.0 / (1.0 + min(max_sample_count, current_sample_count)));
    //float3 res = lerp(clamped_history.rgb, center.rgb, 1.0 / 32);

    const float output_sample_count = min(current_sample_count, max_sample_count) + 1;
    float4 output = working_to_linear(float4(res, output_sample_count));
    history_output_tex[px] = output;

    //output = smoothstep(1.0, 3.0, history_dist);
    //output = abs(history.rgb - ex.rgb);
    //output = dev.rgb;

    //output *= reproj.z;    // debug validity
    //output *= light_stability;
    //output = smoothstep(0.0, 0.05, history_dist);
    //output = length(dev.rgb);
    //output = 1-light_stability;
    //output = control_variate_luma;
    //output = abs(rdiff);
    //output = abs(dev.rgb);
    //output = abs(hist_dev.rgb);
    //output = smoothed_dev;

    // TODO: adaptively sample according to abs(res)
    //output = abs(res);
    //output = WaveActiveSum(center.rgb) / WaveActiveSum(1);
    //output = WaveActiveSum(history.rgb) / WaveActiveSum(1);
    //output.rgb = 0.1 * temporal_change / max(1e-5, working_luma(dev.rgb));
    //output = pow(smoothstep(0.1, 1, temporal_change), 1.0);
    //output.rgb = center_temporal_dev;
    //output = center_dev / max(1e-5, center_luma);
    //output = 1 - smoothstep(0.01, 0.6, temporal_change);
    //output = pow(smoothstep(0.02, 0.6, 0.01 * temporal_change / center_temporal_dev), 0.25);
    //output = max_sample_count / 32.0;
    //output.rgb = temporal_change * 0.1;
    //output.rgb = variance_adjusted_temporal_change * 0.1;
    //output.rgb = rt_history_invalidity_tex[px / 2];
    //output.rgb = lerp(output.rgb, rt_invalid, 0.9);
    //output.rgb = lerp(output.rgb, pow(output_sample_count / 32.0, 4), 0.9);
    //output.r = 1-reproj.w;

    output_tex[px] = float4(
        output.rgb,
        saturate(
            output_sample_count
            * lerp(1.0, 0.5, rt_invalid)
            * smoothstep(0.3, 0, temporal_change)
            / 32.0));

    //output_tex[px] = float4(output.rgb, output_sample_count);
    //output_tex[px] = float4(output.rgb, 1.0 - rt_invalid);
}
