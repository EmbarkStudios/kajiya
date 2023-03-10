#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/bindless_textures.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/lights/triangle.hlsl"

#include "bindings.hlsl"
#include "../ircache/bindings.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

DEFINE_WRC_BINDINGS(0)
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
DEFINE_IRCACHE_BINDINGS(2, 3, 4, 5, 6, 7, 8, 9, 10)
[[vk::binding(11)]] RWTexture2D<float4> output_tex;

#include "lookup.hlsl"
#include "../ircache/lookup.hlsl"

static const float SKY_DIST = 1e4;

#define ROUGHNESS_BIAS 0.5
#define USE_IRCACHE 1

static const uint MAX_PATH_LENGTH = 30;
static const uint RUSSIAN_ROULETTE_START_PATH_LENGTH = 3;
static const float MAX_RAY_LENGTH = FLT_MAX;
//static const float MAX_RAY_LENGTH = 5.0;

static const bool USE_SOFT_SHADOWS = !true;
static const bool USE_LIGHTS = true;
static const bool USE_EMISSIVE = true;

float3 sample_environment_light(float3 dir) {
    return atmosphere_default(dir, SUN_DIRECTION);
}

[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    uint rng = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);

    const float2 pixel_center = px + 0.5;
    const float2 uv = pixel_center / DispatchRaysDimensions().xy;

    RayCone ray_cone = RayCone::from_spread_angle(0.03);
    RayDesc outgoing_ray;
    {
        const ViewRayContext view_ray_context = ViewRayContext::from_uv(uv);
        const float3 ray_dir_ws = view_ray_context.ray_dir_ws();

        outgoing_ray = new_ray(
            view_ray_context.ray_origin_ws(), 
            normalize(ray_dir_ws.xyz),
            0.0,
            FLT_MAX
        );
    }

    WrcFarField far_field =
        WrcFarFieldQuery::from_ray(outgoing_ray.Origin, outgoing_ray.Direction)
            .with_query_normal(direction_view_to_world(float3(0, 0, -1)))
            .query();

    if (far_field.is_hit()) {
        outgoing_ray.TMax = far_field.probe_t;
    }

    float3 total_radiance = 0.0.xxx;
    float3 hit_normal_ws = -outgoing_ray.Direction;
    float hit_t = outgoing_ray.TMax;

    // TODO: cone spread angle
    const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
        .with_cone(ray_cone)
        .with_cull_back_faces(false)
        .with_path_length(1)
        .trace(acceleration_structure);

    if (primary_hit.is_hit) {
        hit_t = primary_hit.ray_t;
        GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
        hit_normal_ws = gbuffer.normal;

        gbuffer.roughness = lerp(gbuffer.roughness, 1.0, ROUGHNESS_BIAS);
        const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
        const float3 wo = mul(-outgoing_ray.Direction, tangent_to_world);
        const LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

        // Sun
        float3 sun_radiance = SUN_COLOR;
        if (any(sun_radiance) > 0) {
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
            const float3 light_radiance = select(is_shadowed, 0.0, sun_radiance);
            total_radiance += brdf_value * light_radiance;
        }

        if (USE_EMISSIVE) {
            total_radiance += gbuffer.emissive;
        }

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
                                1e-3,
                                sqrt(dist_to_light2) - 2e-3
                        ));

                    #if 1
                        const float3 bounce_albedo = lerp(gbuffer.albedo, 1.0.xxx, 0.04);
                        const float3 brdf_value = bounce_albedo * to_psa_metric / M_PI;
                    #else
                        const float3 wi = mul(to_light_norm_ws, tangent_to_world);
                        const float3 brdf_value = brdf.evaluate(wo, wi) * to_psa_metric;
                    #endif

                    total_radiance +=
                        select(!is_shadowed, (triangle_light.radiance() * brdf_value / light_sample.pdf.value), 0);
                }
            }

            if (USE_IRCACHE) {
                const float3 gi = IrcacheLookupParams::create(
                    outgoing_ray.Origin,
                    primary_hit.position,
                    gbuffer.normal)
                    .lookup(rng);

                total_radiance += gi * gbuffer.albedo;
            }
        }
    } else {
        if (far_field.is_hit()) {
            total_radiance += far_field.radiance * far_field.inv_pdf;
        } else {
            total_radiance += sky_cube_tex.SampleLevel(sampler_llr, outgoing_ray.Direction, 0).rgb;
        }
    }

    float3 out_value = total_radiance;
    output_tex[px] = float4(out_value, 1);
}
