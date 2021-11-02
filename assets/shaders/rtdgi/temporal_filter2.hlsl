#include "../inc/samplers.hlsl"
#include "../inc/color.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/soft_color_clamp.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float2> variance_history_tex;
[[vk::binding(3)]] Texture2D<float4> reprojection_tex;
[[vk::binding(4)]] RWTexture2D<float4> output_tex;
[[vk::binding(5)]] RWTexture2D<float4> history_output_tex;
[[vk::binding(6)]] RWTexture2D<float2> variance_history_output_tex;
[[vk::binding(7)]] cbuffer _ {
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
    
    float4 center = input_tex[px];
    float4 reproj = reprojection_tex[px];
    float4 history = history_tex[px];

    output_tex[px] = center;
    //return;
    
    // TODO
    const float light_stability = 1;//center.w;

#if 1
	float4 vsum = 0.0.xxxx;
	float4 vsum2 = 0.0.xxxx;
	float wsum = 0.0;
    float hist_diff = 0.0;
    float hist_vsum = 0.0;
    float hist_vsum2 = 0.0;

    float dev_sum = 0.0;

	const int k = 2;
    {for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = (input_tex[px + int2(x, y)]);
            float4 hist_neigh = (history_tex[px + int2(x, y)]);

            float neigh_luma = calculate_luma(neigh.rgb);
            float hist_luma = calculate_luma(hist_neigh.rgb);

			float w = exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));
			vsum += neigh * w;
			vsum2 += neigh * neigh * w;
			wsum += w;

            dev_sum += neigh.a * neigh.a * w;

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
    dev_sum /= wsum;

    const float2 moments_history = variance_history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);
    //const float center_luma = calculate_luma(center.rgb);
    const float center_luma = calculate_luma(center.rgb) + (hist_vsum - calculate_luma(ex.rgb));// - 0.5 * calculate_luma(control_variate.rgb));
    const float2 current_moments = float2(center_luma, center_luma * center_luma);
    variance_history_output_tex[px] = lerp(moments_history, current_moments, 0.25);
    const float center_temporal_dev = sqrt(max(0.0, moments_history.y - moments_history.x * moments_history.x));

    float center_dev = center.a;

    // Spatial-only variance estimate (dev.rgb) has halos around edges (lighting discontinuities)
    
    // Temporal variance estimate with a spatial boost
    // TODO: this version reduces flicker in pica and on skeletons in battle, but has halos in cornell_box
    //dev.rgb = center_dev * dev.rgb / max(1e-8, clamp(calculate_luma(dev.rgb), center_dev * 0.1, center_dev * 3.0));

    // Spatiotemporal variance estimate
    // TODO: this version seems to work best, but needs to take care near sky
    // TODO: also probably needs to be rgb :P
    dev.rgb = sqrt(dev_sum);

    // Temporal variance estimate with spatial colors
    //dev.rgb *= center_dev / max(1e-8, calculate_luma(dev.rgb));

    float3 hist_dev = sqrt(abs(hist_vsum2 - hist_vsum * hist_vsum));
    //dev.rgb *= 0.1 / max(1e-5, clamp(hist_dev, dev.rgb * 0.1, dev.rgb * 10.0));

    //float temporal_change = abs(hist_vsum - calculate_luma(ex.rgb)) / max(1e-8, hist_vsum + calculate_luma(ex.rgb));
    float temporal_change = abs(hist_vsum - calculate_luma(ex.rgb)) / max(1e-8, hist_vsum + calculate_luma(ex.rgb));
    //float temporal_change = 0.1 * abs(hist_vsum - calculate_luma(ex.rgb)) / max(1e-5, calculate_luma(dev.rgb));
    //temporal_change = 0.02 * temporal_change / max(1e-5, calculate_luma(dev.rgb));
    //temporal_change = WaveActiveSum(temporal_change) / WaveActiveSum(1);

    const float n_deviations = 5.0;// * WaveActiveMin(light_stability);
    //dev = max(dev, history * 0.1);
    //dev = min(dev, history * 0.01);
	float4 nmin = center - dev * n_deviations;
	float4 nmax = center + dev * n_deviations;
#endif

#if 0
	nmin = center;
	nmax = center;

	{const int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = (input_tex[px + int2(x, y) * 2]);
			nmin = min(nmin, neigh);
            nmax = max(nmax, neigh);
        }
    }}

    float3 nmid = lerp(nmin.rgb, nmax.rgb, 0.5);
    nmin.rgb = lerp(nmid, nmin.rgb, 3.0);
    nmax.rgb = lerp(nmid, nmax.rgb, 3.0);
#endif

    //const float light_stability = 1.0 - 0.8 * smoothstep(0.1, 0.5, history_dist);
    //const float light_stability = 1.0 - step(0.01, history_dist);
    //const float light_stability = 1;
    //const float light_stability = center.w > 0.0 ? 1.0 : 0.0;

#if 0
	float4 clamped_history = float4(clamp(history.rgb, nmin.rgb, nmax.rgb), history.a);
#else
    float4 clamped_history = float4(
        soft_color_clamp(center.rgb, history.rgb, ex.rgb, 0.5 * dev.rgb),
        history.a
    );
#endif
    /*const float3 history_dist = abs(history.rgb - ex.rgb) / max(0.1, dev.rgb * 0.5);
    const float3 closest_pt = clamp(history.rgb, center.rgb - dev.rgb * 0.5, center.rgb + dev.rgb * 0.5);
    clamped_history = float4(
        lerp(history.rgb, closest_pt, lerp(0.1, 1.0, smoothstep(1.0, 3.0, history_dist))),
        history.a
    );*/

    //clamped_history = history;
    //clamped_history = center;

    const float remapped_temporal_change = smoothstep(0.01, 0.6, temporal_change);
    const float variance_adjusted_temporal_change = smoothstep(0.1, 1.0, 0.05 * temporal_change / center_temporal_dev);

    float max_sample_count = 32;
    max_sample_count = lerp(max_sample_count, 4, variance_adjusted_temporal_change);
    //max_sample_count = lerp(max_sample_count, 1, smoothstep(0.01, 0.6, 10 * temporal_change * (center_dev / max(1e-5, center_luma))));
    max_sample_count *= light_stability;
    //max_sample_count = 16;

    float current_sample_count = history.a;
    
    float3 res = lerp(clamped_history.rgb, center.rgb, 1.0 / (1.0 + min(max_sample_count, current_sample_count)));
    //float3 res = lerp(clamped_history.rgb, center.rgb, 1.0 / 32);

    history_output_tex[px] = float4(res, min(current_sample_count, max_sample_count) + 1);
    float3 output = max(0.0.xxx, res);

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
    //output = 0.01 * temporal_change / max(1e-5, calculate_luma(dev.rgb));
    //output = pow(smoothstep(0.1, 1, temporal_change), 1.0);
    //output = center_temporal_dev;
    //output = center_dev / max(1e-5, center_luma);
    //output = 1 - smoothstep(0.01, 0.6, temporal_change);
    //output = pow(smoothstep(0.02, 0.6, 0.01 * temporal_change / center_temporal_dev), 0.25);
    //output = max_sample_count / 32.0;
    //output = variance_adjusted_temporal_change;

    output_tex[px] = float4(max(0.0.xxx, output), 1.0);
    //output_tex[px] = float4(current_sample_count.xxx / 32, 1.0);
    //history_output_tex[px] = reproj.w;
}
