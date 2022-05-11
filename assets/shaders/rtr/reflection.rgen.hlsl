#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../ircache/bindings.hlsl"
#include "../wrc/bindings.hlsl"
#include "rtr_settings.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
DEFINE_BLUE_NOISE_SAMPLER_BINDINGS(2, 3, 4)
[[vk::binding(5)]] Texture2D<float4> rtdgi_tex;
[[vk::binding(6)]] TextureCube<float4> sky_cube_tex;
DEFINE_IRCACHE_BINDINGS(7, 8, 9, 10, 11, 12, 13, 14, 15)
DEFINE_WRC_BINDINGS(16)
[[vk::binding(17)]] RWTexture2D<float4> out0_tex;
[[vk::binding(18)]] RWTexture2D<float4> out1_tex;
[[vk::binding(19)]] RWTexture2D<float4> out2_tex;
[[vk::binding(20)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

//#define IRCACHE_LOOKUP_KEEP_ALIVE_PROB 0.125
#include "../ircache/lookup.hlsl"
#include "../wrc/lookup.hlsl"

#include "reflection_trace_common.inc.hlsl"

[shader("raygeneration")]
void main() {
    uint2 hi_px_subpixels[4] = {
        uint2(0, 0),
        uint2(1, 1),
        uint2(1, 0),
        uint2(0, 1),
    };

    const uint2 px = DispatchRaysIndex().xy;
    const uint2 hi_px = px * 2 + hi_px_subpixels[frame_constants.frame_index & 3];
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        out0_tex[px] = float4(0.0.xxx, -SKY_DIST);
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);

    float4 gbuffer_packed = gbuffer_tex[hi_px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    gbuffer.roughness = max(gbuffer.roughness, RTR_ROUGHNESS_CLAMP);

    const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);

#if RTR_USE_TIGHTER_RAY_BIAS
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_biased_depth(uv, depth);
    const float3 refl_ray_origin_ws = view_ray_context.biased_secondary_ray_origin_ws_with_normal(gbuffer.normal);
#else
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);
    const float3 refl_ray_origin_ws = view_ray_context.biased_secondary_ray_origin_ws();
#endif

    float3 wo = mul(-view_ray_context.ray_dir_ws(), tangent_to_world);

    // Hack for shading normals facing away from the outgoing ray's direction:
    // We flip the outgoing ray along the shading normal, so that the reflection's curvature
    // continues, albeit at a lower rate.
    if (wo.z < 0.0) {
        wo.z *= -0.25;
        wo = normalize(wo);
    }

    SpecularBrdf specular_brdf;
    specular_brdf.albedo = lerp(0.04, gbuffer.albedo, gbuffer.metalness);
    specular_brdf.roughness = gbuffer.roughness;

    const uint noise_offset = frame_constants.frame_index * (USE_TEMPORAL_JITTER ? 1 : 0);
    uint rng = hash3(uint3(px, noise_offset));

#if 1
    // Note: since this is pre-baked for various SPP, can run into undersampling
    float2 urand = float2(
        blue_noise_sampler(px.x, px.y, noise_offset, 0),
        blue_noise_sampler(px.x, px.y, noise_offset, 1)
    );
#else
    float2 urand = blue_noise_for_pixel(px, noise_offset).xy;
#endif

    const float sampling_bias = SAMPLING_BIAS;
    urand.x = lerp(urand.x, 0.0, sampling_bias);

    BrdfSample brdf_sample = specular_brdf.sample(wo, urand);
    
#if USE_TEMPORAL_JITTER && !USE_GGX_VNDF_SAMPLING
    [loop] for (uint retry_i = 0; retry_i < 4 && !brdf_sample.is_valid(); ++retry_i) {
        urand = float2(
            uint_to_u01_float(hash1_mut(rng)),
            uint_to_u01_float(hash1_mut(rng))
        );
        urand.x = lerp(urand.x, 0.0, sampling_bias);

        brdf_sample = specular_brdf.sample(wo, urand);
    }
#endif

    const float ratio_estimator_factor = normalize(wo + brdf_sample.wi).z;
    //const float ratio_estimator_factor = brdf_sample.pdf;

    if (brdf_sample.is_valid()) {
        //const bool use_short_ray = gbuffer.roughness > 0.55 && USE_SHORT_RAYS_FOR_ROUGH;

        RayDesc outgoing_ray;
        outgoing_ray.Direction = mul(tangent_to_world, brdf_sample.wi);
        outgoing_ray.Origin = refl_ray_origin_ws;
        outgoing_ray.TMin = 0;
        outgoing_ray.TMax = SKY_DIST;

        uint rng = hash2(px);
        RtrTraceResult result = do_the_thing(px, gbuffer.normal, gbuffer.roughness, rng, outgoing_ray);

        const float3 direction_vs = direction_world_to_view(outgoing_ray.Direction);
        const float to_surface_area_measure =
            #if RTR_APPROX_MEASURE_CONVERSION
                1
            #else
                abs(brdf_sample.wi.z * dot(result.hit_normal_vs, -direction_vs))
            #endif
            / max(1e-10, result.hit_t * result.hit_t);

        const float3 hit_vs =
            #if RTR_RAY_HIT_STORED_AS_POSITION
                view_ray_context.ray_hit_vs() +
            #endif
            direction_vs * result.hit_t;

        const float pdf =
            #if RTR_PDF_STORED_WITH_SURFACE_AREA_METRIC
                to_surface_area_measure *
            #endif
            brdf_sample.pdf;

        out0_tex[px] = float4(result.total_radiance, rtr_encode_ratio_estimator_factor_for_fp16(ratio_estimator_factor));
        out1_tex[px] = float4(hit_vs, pdf);
        out2_tex[px] = float4(result.hit_normal_vs, 0);
    } else {
        out0_tex[px] = float4(0.0.xxx, 0);
        out1_tex[px] = 0.0.xxxx;
    }
}
