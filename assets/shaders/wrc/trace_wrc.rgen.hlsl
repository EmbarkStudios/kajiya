#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/bindless_textures.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/mesh.hlsl"
#include "../inc/sh.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../ircache/bindings.hlsl"
#include "wrc_settings.hlsl"


[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] TextureCube<float4> sky_cube_tex;
DEFINE_IRCACHE_BINDINGS(1, 2, 3, 4, 5, 6, 7, 8, 9)
[[vk::binding(10)]] RWTexture2D<float4> radiance_atlas_out_tex;

#define IRCACHE_LOOKUP_DONT_KEEP_ALIVE   // TODO
#include "../ircache/lookup.hlsl"
#include "../inc/sun.hlsl"

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;
static const bool USE_LIGHTS = true;
static const bool USE_EMISSIVE = true;
static const bool USE_IRCACHE = true;
static const bool USE_BLEND_OUTPUT = true;
static const bool USE_FLICKER_SUPPRESSION = true;
static const uint TARGET_SAMPLE_COUNT = 8;

float3 sample_environment_light(float3 dir) {
    return sky_cube_tex.SampleLevel(sampler_llr, dir, 0).rgb;
}

float pack_dist(float x) {
    return x;
}

float unpack_dist(float x) {
    return x;
}

[shader("raygeneration")]
void main() {
    return;

    const uint probe_idx = DispatchRaysIndex().x / (WRC_PROBE_DIMS * WRC_PROBE_DIMS);
    const uint probe_px_idx = DispatchRaysIndex().x % (WRC_PROBE_DIMS * WRC_PROBE_DIMS);
    const uint2 probe_px = uint2(probe_px_idx % WRC_PROBE_DIMS, probe_px_idx / WRC_PROBE_DIMS);
    const uint2 tile = wrc_probe_idx_to_atlas_tile(probe_idx);
    const uint2 atlas_px = tile * WRC_PROBE_DIMS + probe_px;
    const uint3 probe_coord = wrc_probe_idx_to_coord(probe_idx);
    const uint sequence_index = frame_constants.frame_index % TARGET_SAMPLE_COUNT;

    uint rng = hash_combine2(hash1(probe_idx), sequence_index);

    float3 irradiance_sum = 0;
    float valid_sample_count = 0;
    float hit_count = 0;
    float2 hit_dist_wt = 0;
    const uint sample_count = 1;

    for (uint sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        valid_sample_count += 1.0;

        RayDesc outgoing_ray;
        {
            float3 dir;
            if (USE_BLEND_OUTPUT) {
                dir = octa_decode((probe_px + r2_sequence(sequence_index)) / WRC_PROBE_DIMS);
            } else {
                dir = octa_decode((probe_px + 0.5) / WRC_PROBE_DIMS);
            }

            outgoing_ray = new_ray(
                wrc_probe_center(probe_coord),
                dir,
                WRC_MIN_TRACE_DIST,
                FLT_MAX
            );
        }

        // ----

        float roughness_bias = 0.5;

        {
            const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
                .with_cone(RayCone::from_spread_angle(0.03))
                .with_cull_back_faces(false)
                .with_path_length(1)  // +1 because this is indirect light
                .trace(acceleration_structure);

            if (primary_hit.is_hit) {
                const float3 to_light_norm = SUN_DIRECTION;
                
                const bool is_shadowed = rt_is_shadowed(
                    acceleration_structure,
                    new_ray(
                        primary_hit.position,
                        to_light_norm,
                        1e-4,
                        FLT_MAX
                ));

                hit_dist_wt += float2(pack_dist(primary_hit.ray_t), 1);

                GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();

                const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
                const float3 wi = mul(to_light_norm, tangent_to_world);

                float3 wo = mul(-outgoing_ray.Direction, tangent_to_world);

                // Hack for shading normals facing away from the outgoing ray's direction:
                // We flip the outgoing ray along the shading normal, so that the reflection's curvature
                // continues, albeit at a lower rate.
                if (wo.z < 0.0) {
                    wo.z *= -0.25;
                    wo = normalize(wo);
                }

                LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

                if (FIREFLY_SUPPRESSION) {
                    brdf.specular_brdf.roughness = lerp(brdf.specular_brdf.roughness, 1.0, roughness_bias);
                }

                const float3 brdf_value = brdf.evaluate_directional_light(wo, wi);
                const float3 light_radiance = select(is_shadowed, 0.0, SUN_COLOR);
                irradiance_sum += brdf_value * light_radiance * max(0.0, wi.z);

                if (USE_EMISSIVE) {
                    irradiance_sum += gbuffer.emissive;
                }

                if (USE_LIGHTS && frame_constants.triangle_light_count > 0/* && path_length > 0*/) {   // rtr comp
                    const float light_selection_pmf = 1.0 / frame_constants.triangle_light_count;
                    const uint light_idx = hash1_mut(rng) % frame_constants.triangle_light_count;
                    //const float light_selection_pmf = 1;
                    //for (uint light_idx = 0; light_idx < frame_constants.triangle_light_count; light_idx += 1)
                    {
                        const float2 urand = float2(
                            uint_to_u01_float(hash1_mut(rng)),
                            uint_to_u01_float(hash1_mut(rng))
                        );

                        TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
                        LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
                        const float3 shadow_ray_origin = primary_hit.position;
                        const float3 to_light_ws = light_sample.pos - primary_hit.position;
                        const float dist_to_light2 = dot(to_light_ws, to_light_ws);
                        const float3 to_light_norm_ws = to_light_ws * rsqrt(dist_to_light2);

                        const float to_psa_metric =
                            max(0.0, dot(to_light_norm_ws, gbuffer.normal))
                            * max(0.0, dot(to_light_norm_ws, -light_sample.normal))
                            / dist_to_light2;

                        if (to_psa_metric > 0.0) {
                            float3 wi = mul(to_light_norm_ws, tangent_to_world);

                            const bool is_shadowed =
                                rt_is_shadowed(
                                    acceleration_structure,
                                    new_ray(
                                        shadow_ray_origin,
                                        to_light_norm_ws,
                                        1e-3,
                                        sqrt(dist_to_light2) - 2e-3
                                ));

                            irradiance_sum +=
                                select(is_shadowed, 0,
                                    gbuffer.albedo * triangle_light.radiance() * brdf.evaluate(wo, wi) / light_sample.pdf.value * to_psa_metric / light_selection_pmf);
                        }
                    }
                }

                if (USE_IRCACHE) {
                    const uint rank = 0;    // TODO: how the heck...
                    irradiance_sum +=
                        IrcacheLookupParams::create(outgoing_ray.Origin, primary_hit.position, gbuffer.normal)
                            .with_query_rank(rank)
                            .lookup(rng)
                            * gbuffer.albedo;
                }
                
                hit_count += 1.0;
            } else {
                hit_dist_wt += float2(pack_dist(1e5), 1);
                irradiance_sum += sample_environment_light(outgoing_ray.Direction);
            }
        }
    }

    irradiance_sum /= max(1.0, valid_sample_count);
    irradiance_sum = max(0.0, irradiance_sum);

    float avg_dist = unpack_dist(hit_dist_wt.x / max(1, hit_dist_wt.y));

    //radiance_atlas_out_tex[atlas_px] = float4(float2(probe_px + 0.5) / WRC_PROBE_DIMS, 0.0.xx);

    if (USE_BLEND_OUTPUT) {
        float4 prev_value = radiance_atlas_out_tex[atlas_px];
        float4 new_value = float4(irradiance_sum, avg_dist);
        float4 blended_value = lerp(prev_value, new_value, 1.0 / TARGET_SAMPLE_COUNT);

        if (USE_FLICKER_SUPPRESSION) {
            blended_value.rgb = min(blended_value.rgb, prev_value.rgb * 2 + normalize(blended_value.rgb));
        }

        radiance_atlas_out_tex[atlas_px] = blended_value;
    } else {
        radiance_atlas_out_tex[atlas_px] = float4(irradiance_sum, avg_dist);
    }
}
