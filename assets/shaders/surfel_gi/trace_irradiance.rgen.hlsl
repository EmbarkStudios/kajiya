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
#include "../inc/bindless_textures.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/mesh.hlsl"
#include "../inc/sh.hlsl"

#define HEMISPHERE_ONLY 1


[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] ByteAddressBuffer surfel_hash_key_buf;
[[vk::binding(3)]] ByteAddressBuffer surfel_hash_value_buf;
[[vk::binding(4)]] ByteAddressBuffer cell_index_offset_buf;
[[vk::binding(5)]] ByteAddressBuffer surfel_index_buf;
[[vk::binding(6)]] RWStructuredBuffer<float4> surfel_irradiance_buf;
[[vk::binding(7)]] RWStructuredBuffer<float4> surfel_sh_buf;

#include "../inc/sun.hlsl"

#include "lookup.hlsl"

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;
static const bool USE_EMISSIVE = true;
static const bool SAMPLE_SURFELS_AT_LAST_VERTEX = true;
static const uint MAX_PATH_LENGTH = 1;

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
    uint seed = hash_combine2(hash1(surfel_idx), frame_constants.frame_index);

    const Vertex surfel = unpack_vertex(surfel_spatial_buf[surfel_idx]);

    float4 prev_total_radiance_packed = min(surfel_irradiance_buf[surfel_idx], 32);

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
            float2 urand = float2(
                uint_to_u01_float(hash1_mut(seed)),
                uint_to_u01_float(hash1_mut(seed))
            );

            /*float3 dir = uniform_sample_sphere(urand);

            #if HEMISPHERE_ONLY
                if (dot(dir, surfel.normal) < 0) {
                    dir = reflect(dir, surfel.normal);
                }
            #endif
            //float3 dir = normalize(surfel.normal + uniform_sample_sphere(urand));
            */

            BrdfSample brdf_sample = brdf.sample(float3(0, 0, 1), urand);

            outgoing_ray = new_ray(
                surfel.position,
                mul(tangent_to_world, brdf_sample.wi),
                1e-3,
                FLT_MAX
            );
        }

        // ----

        float3 throughput = 1.0.xxx;
        float roughness_bias = 0.0;

        [loop]
        for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
            const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
                .with_cone(RayCone::from_spread_angle(1e2))
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

                if (SAMPLE_SURFELS_AT_LAST_VERTEX && path_length + 1 == MAX_PATH_LENGTH) {
                    irradiance_sum += lookup_surfel_gi(primary_hit.position, gbuffer.normal) * throughput * gbuffer.albedo;
                }

                const float3 urand = float3(
                    uint_to_u01_float(hash1_mut(seed)),
                    uint_to_u01_float(hash1_mut(seed)),
                    uint_to_u01_float(hash1_mut(seed))
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
                if (0 == path_length) {
                    hit_dist_wt += float2(pack_dist(1), 1);
                }

                irradiance_sum += throughput * sample_environment_light(outgoing_ray.Direction);

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

    const float value_mult = HEMISPHERE_ONLY ? 1 : 2;

    //surfel_irradiance_buf[surfel_idx] = float4(0.0.xxx, total_sample_count);
    surfel_irradiance_buf[surfel_idx] = float4(
        lerp(prev_total_radiance_packed.rgb, irradiance_sum, blend_factor_new),
        total_sample_count
    );
    //surfel_sh_buf[surfel_idx * 3 + 0] = lerp(surfel_sh_buf[surfel_idx * 3 + 0], r_values, blend_factor_new);
    //surfel_sh_buf[surfel_idx * 3 + 1] = lerp(surfel_sh_buf[surfel_idx * 3 + 1], g_values, blend_factor_new);
    //surfel_sh_buf[surfel_idx * 3 + 2] = lerp(surfel_sh_buf[surfel_idx * 3 + 2], b_values, blend_factor_new);
}
