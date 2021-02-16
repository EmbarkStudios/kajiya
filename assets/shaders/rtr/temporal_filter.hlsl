// TODO: currently a copy-pasta of the SSGI filter

#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"

#define USE_DUAL_REPROJECTION 1

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float> depth_tex;
[[vk::binding(3)]] Texture2D<float> ray_len_tex;
[[vk::binding(4)]] Texture2D<float4> reprojection_tex;
[[vk::binding(5)]] RWTexture2D<float4> output_tex;
[[vk::binding(6)]] cbuffer _ {
    float4 output_tex_size;
};
SamplerState sampler_lnc;

//#define LINEAR_TO_WORKING(x) sqrt(x)
//#define WORKING_TO_LINEAR(x) ((x)*(x))

#define LINEAR_TO_WORKING(x) x
#define WORKING_TO_LINEAR(x) x

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    #if 0
        output_tex[px] = float4(ray_len_tex[px].xxx * 0.1, 1);
        return;
    #elif 0
        output_tex[px] = input_tex[px];
        return;
    #endif

    const float4 center = WORKING_TO_LINEAR(max(0.0.xxxx, input_tex[px]));

    float refl_ray_length = clamp(ray_len_tex[px], 0, 1e3);

    // TODO: run a small edge-aware soft-min filter of ray length.
    // The `WaveActiveMin` below improves flat rough surfaces, but is not correct across discontinuities.
    refl_ray_length = WaveActiveMin(refl_ray_length);
    
    float2 uv = get_uv(px, output_tex_size);
    
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth_tex[px]);
    const float3 reflector_vs = view_ray_context.ray_hit_vs();
    const float3 reflection_hit_vs = reflector_vs + view_ray_context.ray_dir_vs() * refl_ray_length;

    const float4 reflection_hit_cs = mul(frame_constants.view_constants.view_to_clip, float4(reflection_hit_vs, 1));
    const float4 prev_hit_cs = mul(frame_constants.view_constants.clip_to_prev_clip, reflection_hit_cs);
    const float2 hit_prev_uv = cs_to_uv(prev_hit_cs.xy / prev_hit_cs.w);

    float4 reproj = reprojection_tex[px];

    float4 history0 = history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);
    float4 history1 = history_tex.SampleLevel(sampler_lnc, hit_prev_uv, 0);

    history0 = WORKING_TO_LINEAR(max(0.0.xxxx, history0));
    history1 = WORKING_TO_LINEAR(max(0.0.xxxx, history1));

	float4 vsum = 0.0.xxxx;
	float4 vsum2 = 0.0.xxxx;
	float wsum = 0.0;

	const int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = WORKING_TO_LINEAR(max(0.0.xxxx, input_tex[px + int2(x, y) * 1]));
			float w = exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));
			vsum += neigh * w;
			vsum2 += neigh * neigh * w;
			wsum += w;
        }
    }

	float4 ex = vsum / wsum;
	float4 ex2 = vsum2 / wsum;
	float4 dev = sqrt(max(0.0.xxxx, ex2 - ex * ex));

    float box_size = 1;

    const float n_deviations = 5;
	float4 nmin = lerp(center, ex, box_size * box_size) - dev * box_size * n_deviations;
	float4 nmax = lerp(center, ex, box_size * box_size) + dev * box_size * n_deviations;
    
    float h0diff = length(history0.xyz - ex.xyz);
    float h1diff = length(history1.xyz - ex.xyz);
    float hdiff_scl = max(1e-10, max(h0diff, h1diff));

#if USE_DUAL_REPROJECTION
    float h0_score = exp2(-100 * min(1, h0diff / hdiff_scl));
    float h1_score = exp2(-100 * min(1, h1diff / hdiff_scl));
#else
    float h0_score = 1;
    float h1_score = 0;
#endif

    const float score_sum = h0_score + h1_score;
    h0_score /= score_sum;
    h1_score /= score_sum;

    float4 clamped_history = clamp(history0 * h0_score + history1 * h1_score, nmin, nmax);
    //float4 clamped_history = clamped_history0 * h0_score + clamped_history1 * h1_score;

    //clamped_history = history0;
    //clamped_history.w = history.w;

    float target_sample_count = 24;//lerp(8, 24, saturate(0.3 * center.w));
    float4 res = lerp(clamped_history, center, lerp(1.0, 1.0 / target_sample_count, reproj.z));
    
    output_tex[px] = LINEAR_TO_WORKING(res);
    //output_tex[px].w = h0_score / (h0_score + h1_score);
    //output_tex[px] = reproj.w;
}
