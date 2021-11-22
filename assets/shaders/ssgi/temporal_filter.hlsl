#include "../inc/uv.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> reprojection_tex;
[[vk::binding(3)]] RWTexture2D<float4> output_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 output_tex_size;
};
SamplerState sampler_lnc;

#define USE_AO_ONLY 1

#define LINEAR_TO_WORKING(x) x
#define WORKING_TO_LINEAR(x) x

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    
    float4 center = WORKING_TO_LINEAR(input_tex[px]);
    float4 reproj = reprojection_tex[px];
    float4 history = WORKING_TO_LINEAR(history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0));
    
	float4 vsum = 0.0.xxxx;
	float4 vsum2 = 0.0.xxxx;
	float wsum = 0.0;

	const int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = WORKING_TO_LINEAR(input_tex[px + int2(x, y) * 2]);
			float w = exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));
			vsum += neigh * w;
			vsum2 += neigh * neigh * w;
			wsum += w;
        }
    }

	float4 ex = vsum / wsum;
	float4 ex2 = vsum2 / wsum;
	float4 dev = sqrt(max(0.0.xxxx, ex2 - ex * ex));

    float box_size = lerp(0.05, 1.0, reproj.w);

    const float n_deviations = 5.0;
	float4 nmin = lerp(center, ex, box_size * box_size) - dev * box_size * n_deviations;
	float4 nmax = lerp(center, ex, box_size * box_size) + dev * box_size * n_deviations;
    
	float4 clamped_history = clamp(history, nmin, nmax);
    float4 res = lerp(clamped_history, center, lerp(1.0, 1.0 / 12.0, reproj.z));

    #if USE_AO_ONLY
        res = res.r;
    #endif
    
    output_tex[px] = LINEAR_TO_WORKING(res);
}
