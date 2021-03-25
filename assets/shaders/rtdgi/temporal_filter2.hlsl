#include "../inc/samplers.hlsl"
#include "../inc/color.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> reprojection_tex;
[[vk::binding(3)]] RWTexture2D<float4> history_output_tex;
[[vk::binding(4)]] RWTexture2D<float4> output_tex;
[[vk::binding(5)]] cbuffer _ {
    float4 output_tex_size;
    float4 gbuffer_tex_size;
};


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    
    float4 center = input_tex[px];
    float4 reproj = reprojection_tex[px];
    float4 history = history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);

    //output_tex[px] = center;
    //return;
    
#if 1
	float4 vsum = 0.0.xxxx;
	float4 vsum2 = 0.0.xxxx;
	float wsum = 0.0;

	const int k = 1;
    {for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = (input_tex[px + int2(x, y)]);
			float w = exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));
			vsum += neigh * w;
			vsum2 += neigh * neigh * w;
			wsum += w;
        }
    }}

	float4 ex = vsum / wsum;
	float4 ex2 = vsum2 / wsum;
	float4 dev = sqrt(max(0.0.xxxx, ex2 - ex * ex));

    float box_size = 1;//lerp(reproj.w, 1.0, 0.5);

    const float light_stability = center.w;

    const float n_deviations = 4.0 * WaveActiveMin(light_stability);
	float4 nmin = lerp(center, ex, box_size * box_size) - dev * box_size * n_deviations;
	float4 nmax = lerp(center, ex, box_size * box_size) + dev * box_size * n_deviations;
#else
	float4 nmin = center;
	float4 nmax = center;
    const float light_stability = center.w;

	const int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = (input_tex[px + int2(x, y) * 2]);
			nmin = min(nmin, neigh);
            nmax = max(nmax, neigh);
        }
    }
#endif

    //const float light_stability = 1.0 - 0.8 * smoothstep(0.1, 0.5, history_dist);
    //const float light_stability = 1.0 - step(0.01, history_dist);
    //const float light_stability = 1;
    //const float light_stability = center.w > 0.0 ? 1.0 : 0.0;

    float reproj_validity_dilated = reproj.z;
    #if 0
        {
         	const int k = 2;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    reproj_validity_dilated = min(reproj_validity_dilated, reprojection_tex[px  + 2 * int2(x, y)].z);
                }
            }
        }
    #else
        reproj_validity_dilated = min(reproj_validity_dilated, WaveReadLaneAt(reproj_validity_dilated, WaveGetLaneIndex() ^ 1));
        reproj_validity_dilated = min(reproj_validity_dilated, WaveReadLaneAt(reproj_validity_dilated, WaveGetLaneIndex() ^ 8));
    #endif

	float4 clamped_history = clamp(history, nmin, nmax);
    //clamped_history = center;
    //clamped_history = history;

    float3 res = lerp(clamped_history.rgb, center.rgb, 1.0 / lerp(1.0, 16.0, reproj_validity_dilated * light_stability));

    history_output_tex[px] = float4(res, 1);

    float3 spatial_input = max(0.0.xxx, res);

    //spatial_input *= reproj.z;    // debug validity
    //spatial_input *= light_stability;
    //spatial_input = smoothstep(0.0, 0.05, history_dist);
    //spatial_input = length(dev.rgb);
    //spatial_input = 1-light_stability;
    //spatial_input = control_variate_luma;
    //spatial_input = abs(rdiff);
    //spatial_input = abs(dev.rgb);
    //spatial_input = smoothed_dev;

    // TODO: adaptively sample according to abs(res)
    //spatial_input = abs(res);

    output_tex[px] = float4(spatial_input, 1.0);
    //history_output_tex[px] = reproj.w;
}
