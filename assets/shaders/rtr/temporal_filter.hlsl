#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/color.hlsl"
#include "../inc/bilinear.hlsl"
#include "rtr_settings.hlsl"

#define USE_DUAL_REPROJECTION 1

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float> depth_tex;
[[vk::binding(3)]] Texture2D<float2> ray_len_tex;
[[vk::binding(4)]] Texture2D<float4> reprojection_tex;
[[vk::binding(5)]] RWTexture2D<float4> output_tex;
[[vk::binding(6)]] cbuffer _ {
    float4 output_tex_size;
};

// Should probably be 3 or 4
#define ENCODING_SCHEME 4

#if 0 == ENCODING_SCHEME
float4 linear_to_working(float4 x) {
    return sqrt(x);
}
float4 working_to_linear(float4 x) {
    return ((x)*(x));
}
#endif

#if 1 == ENCODING_SCHEME
float4 linear_to_working(float4 v) {
    return log(1+sqrt(v));
}
float4 working_to_linear(float4 v) {
    v = exp(v) - 1.0;
    return v * v;
}
#endif

#if 2 == ENCODING_SCHEME
float4 linear_to_working(float4 x) {
    return x;
}
float4 working_to_linear(float4 x) {
    return x;
}
#endif

#if 3 == ENCODING_SCHEME
float4 linear_to_working(float4 v) {
    return float4(ycbcr_to_rgb(v.rgb), v.a);
}
float4 working_to_linear(float4 v) {
    return float4(rgb_to_ycbcr(v.rgb), v.a);
}
#endif

// Strong suppression; reduces noise in very difficult cases but introduces a lot of bias
#if 4 == ENCODING_SCHEME
float4 linear_to_working(float4 v) {
    v.rgb = sqrt(max(0.0, v.rgb));
    v.rgb = rgb_to_ycbcr(v.rgb);
    return v;
}
float4 working_to_linear(float4 v) {
    v.rgb = ycbcr_to_rgb(v.rgb);
    v.rgb *= v.rgb;
    return v;
}
#endif

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    #if 0
        output_tex[px] = float4(ray_len_tex[px].xxx * 0.01, 1);
        //output_tex[px] = float4(ray_len_tex[px].yyy * 0.01, 1);
        return;
    #elif !RTR_USE_TEMPORAL_FILTERS
        output_tex[px] = input_tex[px];
        return;
    #endif

    const float4 center = linear_to_working(input_tex[px]);

    float refl_ray_length = clamp(ray_len_tex[px].y, 0, 1e3);

    // TODO: run a small edge-aware soft-min filter of ray length.
    // The `WaveActiveMin` below improves flat rough surfaces, but is not correct across discontinuities.
    //refl_ray_length = WaveActiveMin(refl_ray_length);
    
    float2 uv = get_uv(px, output_tex_size);
    
    const float center_depth = depth_tex[px];
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, center_depth);
    const float3 reflector_vs = view_ray_context.ray_hit_vs();
    const float3 reflection_hit_vs = reflector_vs + view_ray_context.ray_dir_vs() * refl_ray_length;

    const float4 reflection_hit_cs = mul(frame_constants.view_constants.view_to_sample, float4(reflection_hit_vs, 1));
    const float4 prev_hit_cs = mul(frame_constants.view_constants.clip_to_prev_clip, reflection_hit_cs);
    float2 hit_prev_uv = cs_to_uv(prev_hit_cs.xy / prev_hit_cs.w);

    const float4 prev_reflector_cs = mul(frame_constants.view_constants.clip_to_prev_clip, view_ray_context.ray_hit_cs);
    const float2 reflector_prev_uv = cs_to_uv(prev_reflector_cs.xy / prev_reflector_cs.w);

    float4 reproj = reprojection_tex[px];

    const float2 reflector_move_rate = min(1.0, length(reproj.xy) / length(reflector_prev_uv - uv));
    hit_prev_uv = lerp(uv, hit_prev_uv, reflector_move_rate);

    const uint quad_reproj_valid_packed = uint(reproj.z * 15.0 + 0.5);
    const float4 quad_reproj_valid = (quad_reproj_valid_packed & uint4(1, 2, 4, 8)) != 0;

    float4 history0 = 0.0;
    float history0_valid = 1;
    #if 0
        history0 = linear_to_working(history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0));
    #else
        if (0 == quad_reproj_valid_packed) {
            // Everything invalid
            history0_valid = 0;
        } else if (15 == quad_reproj_valid_packed) {
            history0 = history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);
        } else {
            float4 quad_reproj_valid = (quad_reproj_valid_packed & uint4(1, 2, 4, 8)) != 0;

            const Bilinear bilinear = get_bilinear_filter(uv + reproj.xy, output_tex_size.xy);
            float4 s00 = history_tex[int2(bilinear.origin) + int2(0, 0)];
            float4 s10 = history_tex[int2(bilinear.origin) + int2(1, 0)];
            float4 s01 = history_tex[int2(bilinear.origin) + int2(0, 1)];
            float4 s11 = history_tex[int2(bilinear.origin) + int2(1, 1)];
            float4 weights = get_bilinear_custom_weights(bilinear, quad_reproj_valid);

            if (dot(weights, 1.0) > 1e-5) {
                history0 = apply_bilinear_custom_weights(s00, s10, s01, s11, weights);
            } else {
                // Invalid, but we have to return something.
                history0 = (s00 + s10 + s01 + s11) / 4;
            }
        }
        history0 = linear_to_working(history0);
    #endif


    float4 history1 = linear_to_working(history_tex.SampleLevel(sampler_lnc, hit_prev_uv, 0));
    float history1_valid = 1;

    float4 history0_reproj = reprojection_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);
    float4 history1_reproj = reprojection_tex.SampleLevel(sampler_lnc, hit_prev_uv, 0);

	float4 vsum = 0.0.xxxx;
	float wsum = 0.0;

	const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const int2 sample_px = px + int2(x, y);
            const float sample_depth = depth_tex[sample_px];

            float4 neigh = linear_to_working(input_tex[sample_px]);
			float w = exp2(-200.0 * abs(/*center_normal_vs.z **/ (center_depth / sample_depth - 1.0)));

			vsum += neigh * w;
			wsum += w;
        }
    }

	float4 ex = vsum / wsum;
    float reproj_validity_dilated = reproj.z;
    
    float h0diff = length(history0.xyz - ex.xyz);
    float h1diff = length(history1.xyz - ex.xyz);
    float hdiff_scl = max(1e-10, max(h0diff, h1diff));

#if USE_DUAL_REPROJECTION
    float h0_score = exp2(-100 * min(1, h0diff / hdiff_scl)) * history0_valid;
    float h1_score = exp2(-100 * min(1, h1diff / hdiff_scl)) * history1_valid;
#else
    float h0_score = 0;
    float h1_score = 1;
#endif

    const float score_sum = h0_score + h1_score;
    if (score_sum > 1e-50) {
        h0_score /= score_sum;
        h1_score /= score_sum;
    } else {
        h0_score = 1;
        h1_score = 0;
    }

    float4 history = history0 * h0_score + history1 * h1_score;

    float target_sample_count = 4;
    float4 res = lerp(history, center, lerp(1.0, 1.0 / target_sample_count, reproj_validity_dilated));
    res = working_to_linear(res);

    output_tex[px] = max(0.0.xxxx, res);
}
