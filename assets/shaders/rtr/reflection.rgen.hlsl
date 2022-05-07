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

#define USE_SOFT_SHADOWS 1
#define USE_TEMPORAL_JITTER 1

// Should be off then iterating on reflections,
// but might be a good idea to enable for shipping anything.
#define USE_HEAVY_BIAS 1

#define USE_WORLD_RADIANCE_CACHE 0
#define LOWEST_ROUGHNESS_FOR_RADIANCE_CACHE 0.5

#define USE_IRCACHE 1

// Note: should be off when using dedicated specular lighting passes in addition to RTR
#define USE_EMISSIVE 1
#define USE_LIGHTS 1

// Debug bias in sample reuse with position-based hit storage
#define COLOR_CODE_GROUND_SKY_BLACK_WHITE 0

// Strongly reduces roughness of secondary hits
#define USE_AGGRESSIVE_SECONDARY_ROUGHNESS_BIAS 1

// BRDF bias
#define SAMPLING_BIAS 0.05

#define USE_SCREEN_GI_REPROJECTION 1


#if USE_HEAVY_BIAS
    #undef USE_AGGRESSIVE_SECONDARY_ROUGHNESS_BIAS
    #define USE_AGGRESSIVE_SECONDARY_ROUGHNESS_BIAS 1

    #undef SAMPLING_BIAS
    #define SAMPLING_BIAS 0.15
#endif

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

// Large enough to mean "far away" and small enough so that
// the hit points/vectors fit within fp16.
static const float SKY_DIST = 1e4;

struct RtrTraceResult {
    float3 total_radiance;
    float hit_t;
    float3 hit_normal_vs;
};

RtrTraceResult do_the_thing(uint2 px, float3 normal_ws, float roughness, inout uint rng, RayDesc outgoing_ray) {
#if USE_AGGRESSIVE_SECONDARY_ROUGHNESS_BIAS
    const float roughness_bias = lerp(roughness, 1.0, 0.15);
#else
    const float roughness_bias = 0.5 * roughness;
#endif

    WrcFarField far_field = WrcFarField::create_miss();
    if (USE_WORLD_RADIANCE_CACHE &&
        roughness * (1.0 + 0.15 * uint_to_u01_float(hash1_mut(rng))) > LOWEST_ROUGHNESS_FOR_RADIANCE_CACHE) {
        far_field =
            WrcFarFieldQuery::from_ray(outgoing_ray.Origin, outgoing_ray.Direction)
                .with_interpolation_urand(float3(
                    uint_to_u01_float(hash1_mut(rng)),
                    uint_to_u01_float(hash1_mut(rng)),
                    uint_to_u01_float(hash1_mut(rng))
                ))
                .with_query_normal(normal_ws)
                .query();
    }

    if (far_field.is_hit()) {
        outgoing_ray.TMax = far_field.probe_t;
    }

    // See note in `assets/shaders/rtr/resolve.hlsl`
    const float reflected_cone_spread_angle = sqrt(roughness) * 0.1;

    const RayCone ray_cone =
        pixel_ray_cone_from_image_height(gbuffer_tex_size.y * 0.5)
        .propagate(reflected_cone_spread_angle, length(outgoing_ray.Origin - get_eye_position()));

    // TODO: cone spread angle
    const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
        .with_cone(ray_cone)
        .with_cull_back_faces(false)
        .with_path_length(1)
        .trace(acceleration_structure);

    if (primary_hit.is_hit) {
        GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
        gbuffer.roughness = lerp(gbuffer.roughness, 1.0, roughness_bias);
        const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
        const float3 wo = mul(-outgoing_ray.Direction, tangent_to_world);
        const LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

        // Project the sample into clip space, and check if it's on-screen
        const float3 primary_hit_cs = position_world_to_sample(primary_hit.position);
        const float2 primary_hit_uv = cs_to_uv(primary_hit_cs.xy);
        const float primary_hit_screen_depth = depth_tex.SampleLevel(sampler_nnc, primary_hit_uv, 0);
        const GbufferDataPacked primary_hit_screen_gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex[int2(primary_hit_uv * gbuffer_tex_size.xy)]));
        const float3 primary_hit_screen_normal_ws = primary_hit_screen_gbuffer.unpack_normal();
        const bool is_on_screen =
            all(abs(primary_hit_cs.xy) < 1.0) &&
            inverse_depth_relative_diff(primary_hit_cs.z, primary_hit_screen_depth) < 5e-3 &&
            dot(primary_hit_screen_normal_ws, -outgoing_ray.Direction) > 0.0 &&
            dot(primary_hit_screen_normal_ws, gbuffer.normal) > 0.7;

        float3 total_radiance = 0.0.xxx;
        float3 reflected_normal_vs;
        {
            // Sun
            {
                const float3 to_light_norm = sample_sun_direction(
                    blue_noise_for_pixel(px, frame_constants.frame_index).xy,
                    USE_SOFT_SHADOWS
                );

                const bool is_shadowed =
                    rt_is_shadowed(
                        acceleration_structure,
                        new_ray(
                            primary_hit.position,
                            to_light_norm,
                            1e-4,
                            SKY_DIST
                    ));

                const float3 wi = mul(to_light_norm, tangent_to_world);

                const float3 brdf_value = brdf.evaluate(wo, wi) * max(0.0, wi.z);
                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
                total_radiance += brdf_value * light_radiance;
            }

            reflected_normal_vs = direction_world_to_view(gbuffer.normal);

            if (USE_EMISSIVE) {
                total_radiance += gbuffer.emissive;
            }

            if (USE_SCREEN_GI_REPROJECTION && is_on_screen) {
                const float3 reprojected_radiance =
                    rtdgi_tex.SampleLevel(sampler_nnc, primary_hit_uv, 0).rgb
                    * frame_constants.pre_exposure_delta;

                total_radiance += reprojected_radiance.rgb * gbuffer.albedo;
            } else {
                if (USE_LIGHTS) {
                    float2 urand = float2(
                        uint_to_u01_float(hash1_mut(rng)),
                        uint_to_u01_float(hash1_mut(rng))
                    );

                    for (uint light_idx = 0; light_idx < frame_constants.triangle_light_count; light_idx += 1) {
                        TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
                        LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
                        const float3 shadow_ray_origin = primary_hit.position;
                        const float3 to_light_ws = light_sample.pos - shadow_ray_origin;
                        const float dist_to_light2 = dot(to_light_ws, to_light_ws);
                        const float3 to_light_norm_ws = to_light_ws * rsqrt(dist_to_light2);

                        const float to_psa_metric =
                            max(0.0, dot(to_light_norm_ws, gbuffer.normal))
                            * max(0.0, dot(to_light_norm_ws, -light_sample.normal))
                            / dist_to_light2;

                        if (to_psa_metric > 0.0) {
                            const bool is_shadowed =
                                rt_is_shadowed(
                                    acceleration_structure,
                                    new_ray(
                                        shadow_ray_origin,
                                        to_light_norm_ws,
                                        1e-4,
                                        sqrt(dist_to_light2) - 2e-4
                                ));

                            #if 1
                                const float3 bounce_albedo = lerp(gbuffer.albedo, 1.0.xxx, 0.04);
                                const float3 brdf_value = bounce_albedo * to_psa_metric / M_PI;
                            #else
                                const float3 wi = mul(to_light_norm_ws, tangent_to_world);
                                const float3 brdf_value = brdf.evaluate(wo, wi) * to_psa_metric;
                            #endif

                            total_radiance +=
                                !is_shadowed ? (triangle_light.radiance() * brdf_value / light_sample.pdf.value) : 0;
                        }
                    }
                }

                if (USE_IRCACHE) {
                    float3 gi = lookup_irradiance_cache(
                        outgoing_ray.Origin,
                        primary_hit.position,
                        gbuffer.normal,
                        1,
                        rng
                    );

                    total_radiance += gi * gbuffer.albedo;
                }
           }
        }

        RtrTraceResult result;

        #if COLOR_CODE_GROUND_SKY_BLACK_WHITE
            result.total_radiance = 0.0.xxx;
        #else
            result.total_radiance = total_radiance;
        #endif

        result.hit_t = primary_hit.ray_t;
        result.hit_normal_vs = reflected_normal_vs;
        
        return result;
    } else {
        RtrTraceResult result;

        float hit_t = SKY_DIST;
        float3 far_gi;

        if (far_field.is_hit()) {
            far_gi = far_field.radiance * far_field.inv_pdf;
            hit_t = far_field.approx_surface_t;
        } else {
            far_gi = sky_cube_tex.SampleLevel(sampler_llr, outgoing_ray.Direction, 0).rgb;
        }

        #if COLOR_CODE_GROUND_SKY_BLACK_WHITE
            result.total_radiance = 2.0.xxx;
        #else
            result.total_radiance = far_gi;
        #endif

        result.hit_t = hit_t;
        result.hit_normal_vs = -direction_world_to_view(outgoing_ray.Direction);

        return result;
    }
}

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

        out0_tex[px] = float4(result.total_radiance, ratio_estimator_factor);
        out1_tex[px] = float4(hit_vs, pdf);
        out2_tex[px] = float4(result.hit_normal_vs, 0);
    } else {
        out0_tex[px] = float4(0.0.xxx, 0);
        out1_tex[px] = 0.0.xxxx;
    }
}
