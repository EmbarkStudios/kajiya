#include "../inc/samplers.hlsl"
#include "../inc/color.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/soft_color_clamp.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float2> variance_history_tex;
[[vk::binding(3)]] Texture2D<float4> reprojection_tex;
[[vk::binding(4)]] RWTexture2D<float4> history_output_tex;
[[vk::binding(5)]] RWTexture2D<float2> variance_history_output_tex;
[[vk::binding(6)]] RWTexture2D<float4> output_tex;
[[vk::binding(7)]] cbuffer _ {
    float4 output_tex_size;
    float4 gbuffer_tex_size;
};


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    #if 0
        output_tex[px] = max(0.0, input_tex[px]);
        history_output_tex[px] = float4(max(0.0, input_tex[px].rgb), 32);
        return;
    #endif

    float2 uv = get_uv(px, output_tex_size);
    
    float4 center = input_tex[px];
    float4 reproj = reprojection_tex[px];
    float4 history = history_tex[px];

    output_tex[px] = center;
    
    // TODO
    const float light_stability = 1;//center.w;

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
            float4 neigh = (input_tex[px + int2(x, y) * 2]);
            float4 hist_neigh = (history_tex[px + int2(x, y) * 2]);

            float neigh_luma = calculate_luma(neigh.rgb);
            float hist_luma = calculate_luma(hist_neigh.rgb);

			float w = exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));
			vsum += neigh * w;
			vsum2 += neigh * neigh * w;
			wsum += w;

            dev_sum += neigh.a * neigh.a * w;

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
    const float center_luma = calculate_luma(center.rgb) + (hist_vsum - calculate_luma(ex.rgb));
    const float2 current_moments = float2(center_luma, center_luma * center_luma);
    variance_history_output_tex[px] = lerp(moments_history, current_moments, 0.25);
    const float center_temporal_dev = sqrt(max(0.0, moments_history.y - moments_history.x * moments_history.x));

    float center_dev = center.a;

    // Spatial-only variance estimate (dev.rgb) has halos around edges (lighting discontinuities)
    
    // Spatiotemporal variance estimate
    dev.rgb = sqrt(dev_sum);

    // Temporal variance estimate with spatial colors
    float3 hist_dev = sqrt(abs(hist_vsum2 - hist_vsum * hist_vsum));

    float temporal_change = abs(hist_vsum - calculate_luma(ex.rgb)) / max(1e-8, hist_vsum + calculate_luma(ex.rgb));

    const float n_deviations = 5.0;
	float4 nmin = center - dev * n_deviations;
	float4 nmax = center + dev * n_deviations;

    float4 clamped_history = float4(
        soft_color_clamp(center.rgb, history.rgb, ex.rgb, dev.rgb),
        history.a
    );

    const float remapped_temporal_change = smoothstep(0.01, 0.6, temporal_change);
    const float variance_adjusted_temporal_change = smoothstep(0.1, 1.0, 0.05 * temporal_change / center_temporal_dev);

    float max_sample_count = 32;
    max_sample_count = lerp(max_sample_count, 4, variance_adjusted_temporal_change);
    max_sample_count *= light_stability;

    float current_sample_count = history.a;
    
    float3 res = lerp(clamped_history.rgb, center.rgb, 1.0 / (1.0 + min(max_sample_count, current_sample_count)));

    history_output_tex[px] = float4(res, min(current_sample_count, max_sample_count) + 1);
    float3 output = max(0.0.xxx, res);

    output_tex[px] = float4(max(0.0.xxx, output), 1.0);
}
