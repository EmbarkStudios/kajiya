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
#include "near_field_settings.hlsl"
#include "rtdgi_restir_settings.hlsl"
#include "rtdgi_common.hlsl"

[[vk::binding(0)]] Texture2D<float4> radiance_tex;
[[vk::binding(1)]] Texture2D<uint2> reservoir_input_tex;
[[vk::binding(2)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(3)]] Texture2D<float> depth_tex;
[[vk::binding(4)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(5)]] Texture2D<float> half_depth_tex;
[[vk::binding(6)]] Texture2D<float4> ssao_tex;
[[vk::binding(7)]] Texture2D<float4> candidate_radiance_tex;
[[vk::binding(8)]] Texture2D<float4> candidate_hit_tex;
[[vk::binding(9)]] Texture2D<uint4> temporal_reservoir_packed_tex;
[[vk::binding(10)]] Texture2D<float3> bounced_radiance_input_tex;
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
    const int2 hi_px_offset = HALFRES_SUBSAMPLE_OFFSET;

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
    const float center_ssao = ssao_tex[px].r;

    //const float3 center_bent_normal_ws = normalize(direction_view_to_world(ssao_tex[px * 2].gba));

    const uint frame_hash = hash1(frame_constants.frame_index);
    const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + frame_hash) & 3;
    const float4 blue = blue_noise_for_pixel(px, frame_constants.frame_index) * M_TAU;

    const float NEAR_FIELD_FADE_OUT_END = -view_ray_context.ray_hit_vs().z * (SSGI_NEAR_FIELD_RADIUS * output_tex_size.w * 0.5);
    const float NEAR_FIELD_FADE_OUT_START = NEAR_FIELD_FADE_OUT_END * 0.5;

    #if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
        // The near field cannot be fully trusted in tight corners because our irradiance cache
        // has limited resolution, and is likely to create artifacts. Opt on the side of shadowing.
        const float near_field_influence = center_ssao;
    #else
        const float near_field_influence = select(is_rtdgi_tracing_frame(), center_ssao, 0);
    #endif

    float3 total_irradiance = 0;
    bool sharpen_gi_kernel = false;

    {
    float w_sum = 0;
    float3 weighted_irradiance = 0;

    for (uint sample_i = 0; sample_i < select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER, 4, 1); ++sample_i) {
        const float ang = (sample_i + blue.x) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
        const float radius =
            select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER
            , (pow(float(sample_i), 0.666) * 1.0 + 0.4)
            , 0.0);
        const float2 reservoir_px_offset = float2(cos(ang), sin(ang)) * radius;
        const int2 rpx = int2(floor(float2(px) * 0.5 + reservoir_px_offset));

        const float2 rpx_uv = get_uv(
            rpx * 2 + HALFRES_SUBSAMPLE_OFFSET,
            gbuffer_tex_size);
        const float rpx_depth = half_depth_tex[rpx];
        const ViewRayContext rpx_ray_ctx = ViewRayContext::from_uv_and_depth(rpx_uv, rpx_depth);

        if (USE_SPLIT_RT_NEAR_FIELD) {
            const float3 hit_ws = candidate_hit_tex[rpx].xyz + rpx_ray_ctx.ray_hit_ws();
            const float3 sample_offset = hit_ws - view_ray_context.ray_hit_ws();
            const float sample_dist = length(sample_offset);
            const float3 sample_dir = sample_offset / sample_dist;

            const float geometric_term =
                // TODO: fold the 2 into the PDF
                2 * max(0.0, dot(center_normal_ws, sample_dir));

            const float atten = smoothstep(NEAR_FIELD_FADE_OUT_END, NEAR_FIELD_FADE_OUT_START, sample_dist);
            sharpen_gi_kernel |= atten > 0.9;

            float3 contribution = candidate_radiance_tex[rpx].rgb * geometric_term;
            contribution *= lerp(0.0, atten, near_field_influence);

            float3 sample_normal_vs = half_view_normal_tex[rpx].rgb;
            const float sample_ssao = ssao_tex[rpx * 2 + HALFRES_SUBSAMPLE_OFFSET].r;

            float w = 1;
            w *= ggx_ndf_unnorm(0.01, saturate(dot(center_normal_vs, sample_normal_vs)));
            w *= exp2(-200.0 * abs(center_normal_vs.z * (center_depth / rpx_depth - 1.0)));

            weighted_irradiance += contribution * w;
            w_sum += w;
        }
    }

    total_irradiance += weighted_irradiance / max(1e-20, w_sum);
    }

    {
    float w_sum = 0;
    float3 weighted_irradiance = 0;

    const float kernel_scale = select(sharpen_gi_kernel, 0.5, 1.0);
    
    for (uint sample_i = 0; sample_i < select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER, 4, 1); ++sample_i) {
        const float ang = (sample_i + blue.x) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
        const float radius =
            select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER
            , (pow(float(sample_i), 0.666) * 1.0 * kernel_scale + 0.4 * kernel_scale)
            , 0.0);

        const float2 reservoir_px_offset = float2(cos(ang), sin(ang)) * radius;
        const int2 rpx = int2(floor(float2(px) * 0.5 + reservoir_px_offset));

        Reservoir1spp r = Reservoir1spp::from_raw(reservoir_input_tex[rpx]);
        const uint2 spx = reservoir_payload_to_px(r.payload);

        const TemporalReservoirOutput spx_packed = TemporalReservoirOutput::from_raw(temporal_reservoir_packed_tex[spx]);

        const float2 spx_uv = get_uv(
            spx * 2 + HALFRES_SUBSAMPLE_OFFSET,
            gbuffer_tex_size);
        const ViewRayContext spx_ray_ctx = ViewRayContext::from_uv_and_depth(spx_uv, spx_packed.depth);

        {
            const float spx_depth = spx_packed.depth;
            const float rpx_depth = half_depth_tex[rpx];

            const float3 hit_ws = spx_packed.ray_hit_offset_ws + spx_ray_ctx.ray_hit_ws();
            const float3 sample_offset = hit_ws - view_ray_context.ray_hit_ws();
            const float sample_dist = length(sample_offset);
            const float3 sample_dir = sample_offset / sample_dist;

            const float geometric_term =
                // TODO: fold the 2 into the PDF
                2 * max(0.0, dot(center_normal_ws, sample_dir));

            float3 radiance;
            if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
                radiance = bounced_radiance_input_tex[rpx];
            } else {
                radiance = radiance_tex[spx].rgb;
            }

            if (USE_SPLIT_RT_NEAR_FIELD) {
                const float atten = smoothstep(NEAR_FIELD_FADE_OUT_START, NEAR_FIELD_FADE_OUT_END, sample_dist);
                radiance *= lerp(1.0, atten, near_field_influence);
            }

            const float3 contribution = radiance * geometric_term * r.W;

            float3 sample_normal_vs = half_view_normal_tex[spx].rgb;
            const float sample_ssao = ssao_tex[rpx * 2 + HALFRES_SUBSAMPLE_OFFSET].r;

            float w = 1;
            w *= ggx_ndf_unnorm(0.01, saturate(dot(center_normal_vs, sample_normal_vs)));
            w *= exp2(-200.0 * abs(center_normal_vs.z * (center_depth / rpx_depth - 1.0)));
            w *= exp2(-20.0 * abs(center_ssao - sample_ssao));

            weighted_irradiance += contribution * w;
            w_sum += w;
        }
    }
    
    total_irradiance += weighted_irradiance / max(1e-20, w_sum);
    }

    irradiance_output_tex[px] = float4(total_irradiance, 1);
}
