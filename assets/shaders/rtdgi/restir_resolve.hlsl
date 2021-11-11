#include "../inc/color.hlsl"
#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/reservoir.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../rtdgi/near_field_settings.hlsl"
#include "restir_settings.hlsl"
#include "candidate_ray_dir.hlsl"

[[vk::binding(0)]] Texture2D<float4> irradiance_tex;
[[vk::binding(1)]] Texture2D<float4> hit_normal_tex;
[[vk::binding(2)]] Texture2D<float4> ray_tex;
[[vk::binding(3)]] Texture2D<float4> reservoir_input_tex;
[[vk::binding(4)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(5)]] Texture2D<float> depth_tex;
[[vk::binding(6)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(7)]] Texture2D<float> half_depth_tex;
[[vk::binding(8)]] Texture2D<float4> ssao_tex;
[[vk::binding(9)]] Texture2D<float4> candidate_irradiance_tex;
[[vk::binding(10)]] Texture2D<float4> candidate_hit_tex;
[[vk::binding(11)]] RWTexture2D<float4> irradiance_output_tex;
[[vk::binding(12)]] cbuffer _ {
    float4 gbuffer_tex_size;
    float4 output_tex_size;
};

static float ggx_ndf_unnorm(float a2, float cos_theta) {
	float denom_sqrt = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
	return a2 / (denom_sqrt * denom_sqrt);
}

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID) {
    /*const uint2 hi_px_subpixels[4] = {
        uint2(0, 0),
        uint2(1, 1),
        uint2(1, 0),
        uint2(0, 1),
    };
    const uint2 hi_px = px * 2 + hi_px_subpixels[frame_constants.frame_index & 3];*/
    float depth = depth_tex[px];
    if (0 == depth) {
        irradiance_output_tex[px] = 0;
        return;
    }

    const uint seed = frame_constants.frame_index;
    uint rng = hash3(uint3(px, seed));

    const float2 uv = get_uv(px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex[px])).unpack();

    const float3 center_normal_ws = gbuffer.normal;
    const float3 center_normal_vs = direction_world_to_view(center_normal_ws);
    const float center_depth = depth;

    //const float3 center_bent_normal_ws = normalize(direction_view_to_world(ssao_tex[px * 2].gba));

    float3 weighted_irradiance = 0;
    float w_sum = 0;

    const uint frame_hash = hash1(frame_constants.frame_index);
    const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + frame_hash) & 3;
    const float4 blue = blue_noise_for_pixel(px, frame_constants.frame_index) * M_TAU;

    for (uint sample_i = 0; sample_i < 4; ++sample_i) {
        float3 irradiance_sum = 0;

        const float ang = (sample_i + blue.x) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
        const float radius = pow(float(sample_i), 0.666) * 1.0 + 0.4;
        const float2 reservoir_px_offset = float2(cos(ang), sin(ang)) * radius;
        const int2 rpx = int2(floor(float2(px) * 0.5 + reservoir_px_offset));

        Reservoir1spp r = Reservoir1spp::from_raw(reservoir_input_tex[rpx]);
        const uint2 spx = reservoir_payload_to_px(r.payload);

        const float3 hit_ws = ray_tex[spx].xyz;// + get_eye_position();
        const float3 sample_offset = hit_ws - view_ray_context.ray_hit_ws();
        const float sample_dist = length(sample_offset);
        const float3 sample_dir = sample_offset / sample_dist;
        const float3 sample_hit_normal = hit_normal_tex[spx].xyz;

        float geometric_term =
            max(0.0, dot(center_normal_ws, sample_dir))
            // TODO: wtf, why 2
            * (DIFFUSE_GI_SAMPLING_FULL_SPHERE ? M_PI : 2);
        float3 radiance = irradiance_tex[spx].rgb;

        const bool SPLIT = USE_SPLIT_RT_NEAR_FIELD && !USE_SSGI_NEAR_FIELD;
        const float NEAR_FIELD_FADE_OUT_END = -view_ray_context.ray_hit_vs().z * (SSGI_NEAR_FIELD_RADIUS * output_tex_size.w * 0.5);
        const float NEAR_FIELD_FADE_OUT_START = NEAR_FIELD_FADE_OUT_END * 0.5;

        /*if (USE_SSGI_NEAR_FIELD && dot(sample_hit_normal, hit_ws - get_eye_position()) < 0) {
            float infl = sample_dist / (SSGI_NEAR_FIELD_RADIUS * output_tex_size.w * 0.5) / -view_ray_context.ray_hit_vs().z;
            // eyeballed
            radiance *= lerp(0.2, 1.0, smoothstep(0.0, 1.0, infl));
        }*/

        if (SPLIT) {
            radiance *= smoothstep(NEAR_FIELD_FADE_OUT_START, NEAR_FIELD_FADE_OUT_END, sample_dist);
        }

        irradiance_sum += radiance * geometric_term * r.W;

        if (SPLIT) {
            const float hit_t = candidate_hit_tex[rpx].w;
            if (hit_t < NEAR_FIELD_FADE_OUT_END) {
                const float3x3 tangent_to_world = build_orthonormal_basis(center_normal_ws);
                const float3 outgoing_dir_ws = rtdgi_candidate_ray_dir(rpx, tangent_to_world);

                float geometric_term =
                    max(0.0, dot(center_normal_ws, outgoing_dir_ws))
                    * (DIFFUSE_GI_SAMPLING_FULL_SPHERE ? M_PI : 2);
                irradiance_sum += candidate_irradiance_tex[rpx].rgb * geometric_term * smoothstep(NEAR_FIELD_FADE_OUT_END, NEAR_FIELD_FADE_OUT_START, hit_t);
            }
        }

        const float sample_depth = half_depth_tex[rpx];
        const float3 sample_normal_vs = half_view_normal_tex[rpx].rgb;

        float w = 1;
        w *= ggx_ndf_unnorm(0.01, saturate(dot(center_normal_vs, sample_normal_vs)));
        w *= exp2(-200.0 * abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)));

        weighted_irradiance += irradiance_sum * w;
        w_sum += w;
    }

    /*#if DIFFUSE_GI_BRDF_SAMPLING
        irradiance_sum /= max(1e-20, w_sum);
    #endif*/

    weighted_irradiance /= max(1e-20, w_sum);

    irradiance_output_tex[px] = float4(weighted_irradiance, 1);
}
