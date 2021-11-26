#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/bindless_textures.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/mesh.hlsl"
#include "../inc/sh.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../wrc/bindings.hlsl"

#define USE_WORLD_RADIANCE_CACHE 0


[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] ByteAddressBuffer surfel_hash_key_buf;
[[vk::binding(3)]] ByteAddressBuffer surfel_hash_value_buf;
[[vk::binding(4)]] ByteAddressBuffer cell_index_offset_buf;
[[vk::binding(5)]] ByteAddressBuffer surfel_index_buf;
DEFINE_WRC_BINDINGS(6)
[[vk::binding(7)]] RWStructuredBuffer<float4> surfel_irradiance_buf;
[[vk::binding(8)]] RWStructuredBuffer<float4> surfel_sh_buf;

#include "../inc/sun.hlsl"
#include "../wrc/lookup.hlsl"

#include "lookup.hlsl"

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;
static const bool USE_LIGHTS = true;
static const bool USE_EMISSIVE = true;
static const bool SAMPLE_SURFELS_AT_LAST_VERTEX = true;
static const uint MAX_PATH_LENGTH = 1;
static const uint TARGET_SAMPLE_COUNT = 32;

float3 sample_environment_light(float3 dir) {
    //return 0.0.xxx;
    return sky_cube_tex.SampleLevel(sampler_llr, dir, 0).rgb;
    /*return atmosphere_default(dir, SUN_DIRECTION);

    float3 col = (dir.zyx * float3(1, 1, -1) * 0.5 + float3(0.6, 0.5, 0.5)) * 0.75;
    col = lerp(col, 1.3.xxx * calculate_luma(col), smoothstep(-0.2, 1.0, dir.y).xxx);
    return col;*/
}

float pack_dist(float x) {
    return min(1, x);
}

float unpack_dist(float x) {
    return x;
}

[shader("raygeneration")]
void main() {
    const uint surfel_idx = DispatchRaysIndex().x;
    uint rng = hash_combine2(hash1(surfel_idx), frame_constants.frame_index);

    const Vertex surfel = unpack_vertex(surfel_spatial_buf[surfel_idx]);

    float4 prev_total_radiance_packed = min(surfel_irradiance_buf[surfel_idx], TARGET_SAMPLE_COUNT);

    DiffuseBrdf brdf;
    const float3x3 tangent_to_world = build_orthonormal_basis(surfel.normal);

    brdf.albedo = 1.0.xxx;

    const uint sample_count = 4;//clamp(int(32 - prev_total_radiance_packed.w), 1, 32);
    float3 irradiance_sum = 0;
    float valid_sample_count = 0;
    float2 hit_dist_wt = 0;

    for (uint sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        valid_sample_count += 1.0;

        RayDesc outgoing_ray;
        {
            #if 0
                float2 urand = float2(
                    uint_to_u01_float(hash1_mut(rng)),
                    uint_to_u01_float(hash1_mut(rng))
                );
            #else
                float2 urand = r2_sequence(
                    (frame_constants.frame_index * sample_count + sample_idx + hash1(surfel_idx)
                ) % (TARGET_SAMPLE_COUNT * 64));
            #endif

            BrdfSample brdf_sample = brdf.sample(float3(0, 0, 1), urand);

            outgoing_ray = new_ray(
                surfel.position,
                mul(tangent_to_world, brdf_sample.wi),
                0.0,
                FLT_MAX
            );
        }

        #if USE_WORLD_RADIANCE_CACHE
            WrcFarField far_field =
                WrcFarFieldQuery::from_ray(outgoing_ray.Origin, outgoing_ray.Direction)
                    .with_interpolation_urand(float3(
                        uint_to_u01_float(hash1_mut(rng)),
                        uint_to_u01_float(hash1_mut(rng)),
                        uint_to_u01_float(hash1_mut(rng))
                    ))
                    .with_query_normal(surfel.normal)
                    .query();
        #else
            WrcFarField far_field = WrcFarField::create_miss();
        #endif

        if (far_field.is_hit()) {
            outgoing_ray.TMax = far_field.probe_t;
        }

        // ----

        float3 throughput = 1.0.xxx;
        float roughness_bias = 0.0;

        [loop]
        for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
            const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
                .with_cone(RayCone::from_spread_angle(0.5))
                .with_cull_back_faces(true)
                .with_path_length(path_length + 1)  // +1 because this is indirect light
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

                if (0 == path_length) {
                    hit_dist_wt += float2(pack_dist(primary_hit.ray_t), 1);
                }

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
                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
                irradiance_sum += throughput * brdf_value * light_radiance * max(0.0, wi.z);

                if (USE_EMISSIVE) {
                    irradiance_sum += gbuffer.emissive * throughput;
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
                                is_shadowed ? 0 :
                                    throughput * triangle_light.radiance() * brdf.evaluate(wo, wi) / light_sample.pdf.value * to_psa_metric / light_selection_pmf;
                        }
                    }
                }
                
                if (SAMPLE_SURFELS_AT_LAST_VERTEX && path_length + 1 == MAX_PATH_LENGTH) {
                    irradiance_sum += lookup_surfel_gi(primary_hit.position, gbuffer.normal) * throughput * gbuffer.albedo;
                }

                const float3 urand = float3(
                    uint_to_u01_float(hash1_mut(rng)),
                    uint_to_u01_float(hash1_mut(rng)),
                    uint_to_u01_float(hash1_mut(rng))
                );
                BrdfSample brdf_sample = brdf.sample(wo, urand);

                // TODO: investigate NaNs here.
                if (brdf_sample.is_valid() && brdf_sample.value_over_pdf.x == brdf_sample.value_over_pdf.x) {
                    roughness_bias = lerp(roughness_bias, 1.0, 0.5 * brdf_sample.approx_roughness);
                    outgoing_ray.Origin = primary_hit.position;
                    outgoing_ray.Direction = mul(tangent_to_world, brdf_sample.wi);
                    outgoing_ray.TMin = 1e-4;
                    throughput *= brdf_sample.value_over_pdf;
                } else {
                    break;
                }
            } else {
                if (far_field.is_hit()) {
                    irradiance_sum += throughput * far_field.radiance * far_field.inv_pdf;
                } else {
                    if (0 == path_length) {
                        hit_dist_wt += float2(pack_dist(1), 1);
                    }

                    irradiance_sum += throughput * sample_environment_light(outgoing_ray.Direction);
                }

                break;
            }
        }
    }

    irradiance_sum /= max(1.0, valid_sample_count);

    float avg_dist = unpack_dist(hit_dist_wt.x / max(1, hit_dist_wt.y));
    //total_radiance.xyz = lerp(float3(1, 0, 0), float3(0.02, 1.0, 0.2), avg_dist) * total_radiance.w;

    const uint MAX_SAMPLE_COUNT = 64;

    const float total_sample_count = prev_total_radiance_packed.w + valid_sample_count;
    const float blend_factor_new = valid_sample_count / max(1, total_sample_count);
    //const float blend_factor_old = prev_total_radiance_packed.w / total_sample_count;

    //const float contrib_count = prev_total_radiance_packed.w;
    /*if (contrib_count < MAX_SAMPLE_COUNT) {
        surfel_irradiance_buf[surfel_idx] = prev_total_radiance_packed + total_radiance;
    } else {
        surfel_irradiance_buf[surfel_idx] = float4(lerp(
            prev_total_radiance_packed.xyz,
            total_radiance.xyz * MAX_SAMPLE_COUNT,
            0.5 / MAX_SAMPLE_COUNT), MAX_SAMPLE_COUNT);
    }*/

    float3 prev_value = prev_total_radiance_packed.rgb;
    float3 new_value = irradiance_sum;

    float3 blended_value = lerp(prev_value, new_value, blend_factor_new);

    //surfel_irradiance_buf[surfel_idx] = float4(0.0.xxx, total_sample_count);
    surfel_irradiance_buf[surfel_idx] = max(0.0, float4(
        blended_value,
        total_sample_count
    ));
    //surfel_sh_buf[surfel_idx * 3 + 0] = lerp(surfel_sh_buf[surfel_idx * 3 + 0], r_values, blend_factor_new);
    //surfel_sh_buf[surfel_idx * 3 + 1] = lerp(surfel_sh_buf[surfel_idx * 3 + 1], g_values, blend_factor_new);
    //surfel_sh_buf[surfel_idx * 3 + 2] = lerp(surfel_sh_buf[surfel_idx * 3 + 2], b_values, blend_factor_new);
}
