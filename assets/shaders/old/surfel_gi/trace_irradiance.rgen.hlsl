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
#include "directional_basis.hlsl"

#define HEMISPHERE_ONLY 1


[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(1)]] RWStructuredBuffer<float4> surfel_irradiance_buf;
[[vk::binding(2)]] RWStructuredBuffer<float4> surfel_sh_buf;

static const uint MAX_PATH_LENGTH = 5;
#include "../inc/sun.hlsl"


// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;

float3 sample_environment_light(float3 dir) {
    return 0.0.xxx;
    return atmosphere_default(dir, SUN_DIRECTION);

    float3 col = (dir.zyx * float3(1, 1, -1) * 0.5 + float3(0.6, 0.5, 0.5)) * 0.75;
    col = lerp(col, 1.3.xxx * calculate_luma(col), smoothstep(-0.2, 1.0, dir.y).xxx);
    return col;
}


float3 uniform_sample_sphere(float2 urand) {
    float z = 1.0 - 2.0 * urand.x;
    float xy = sqrt(max(0.0, 1.0 - z * z));
    float sn = sin(M_TAU * urand.y);
	float cs = cos(M_TAU * urand.y);
	return float3(cs * xy, sn * xy, z);
}

float3 uniform_sample_hemisphere(float2 urand) {
     float phi = urand.y * M_TAU;
     float cos_theta = 1.0 - urand.x;
     float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
     return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
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

    float4 prev_total_radiance_packed = surfel_irradiance_buf[surfel_idx];

    const uint sample_count = clamp(int(32 - prev_total_radiance_packed.w), 1, 32);
    float valid_sample_count = 0;
    float3 basis_radiance_sums[4] = { 0.0.xxx, 0.0.xxx, 0.0.xxx, 0.0.xxx };

    float2 hit_dist_wt = 0;

    for (uint sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        valid_sample_count += 1.0;

        RayDesc outgoing_ray;
        {
            float2 urand = float2(
                uint_to_u01_float(hash1_mut(seed)),
                uint_to_u01_float(hash1_mut(seed))
            );

            float3 dir = uniform_sample_sphere(urand);

            #if HEMISPHERE_ONLY
                if (dot(dir, surfel.normal) < 0) {
                    dir = reflect(dir, surfel.normal);
                }
            #endif
            //float3 dir = normalize(surfel.normal + uniform_sample_sphere(urand));

            outgoing_ray = new_ray(
                surfel.position,
                normalize(dir),
                1e-3,
                FLT_MAX
            );
        }

        //

        const float3 surfel_tet_basis[4] = calc_surfel_tet_basis(surfel.normal);

        // ----

        float3 throughput = 1.0.xxx;
        float roughness_bias = 0.0;

        [loop]
        for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
            const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
                .with_cone(RayCone::from_spread_angle(1.0))
                .with_cull_back_faces(true)
                .with_path_length(path_length + 1)  // +1 because this is indirect light
                .trace(acceleration_structure);

            if (primary_hit.is_hit) {
                const float3 to_light_norm = SUN_DIRECTION;
                
                const bool is_shadowed = path_length+1 >= MAX_PATH_LENGTH || rt_is_shadowed(
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

                const float3 brdf_value = brdf.evaluate(wo, wi) * max(0.0, wi.z);
                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
                const float3 contrib = throughput * brdf_value * light_radiance;

                [unroll]
                for (int b = 0; b < 4; ++b) {
                    basis_radiance_sums[b] += max(0.0, dot(surfel_tet_basis[b], outgoing_ray.Direction)) * contrib;
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

                const float3 contrib = throughput * sample_environment_light(outgoing_ray.Direction);

                [unroll]
                for (int b = 0; b < 4; ++b) {
                    basis_radiance_sums[b] += max(0.0, dot(surfel_tet_basis[b], outgoing_ray.Direction)) * contrib;
                }

                break;
            }
        }
    }

    [unroll]
    for (int b = 0; b < 4; ++b) {
        basis_radiance_sums[b] /= max(1.0, valid_sample_count);
    }

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

    const float4 r_values = value_mult * float4(basis_radiance_sums[0].r, basis_radiance_sums[1].r, basis_radiance_sums[2].r, basis_radiance_sums[3].r);
    const float4 g_values = value_mult * float4(basis_radiance_sums[0].g, basis_radiance_sums[1].g, basis_radiance_sums[2].g, basis_radiance_sums[3].g);
    const float4 b_values = value_mult * float4(basis_radiance_sums[0].b, basis_radiance_sums[1].b, basis_radiance_sums[2].b, basis_radiance_sums[3].b);

    surfel_irradiance_buf[surfel_idx] = float4(0.0.xxx, total_sample_count);
    //surfel_irradiance_buf[surfel_idx] = float4(total_radiance.xyz + 0.5, total_sample_count);
    surfel_sh_buf[surfel_idx * 3 + 0] = lerp(surfel_sh_buf[surfel_idx * 3 + 0], r_values, blend_factor_new);
    surfel_sh_buf[surfel_idx * 3 + 1] = lerp(surfel_sh_buf[surfel_idx * 3 + 1], g_values, blend_factor_new);
    surfel_sh_buf[surfel_idx * 3 + 2] = lerp(surfel_sh_buf[surfel_idx * 3 + 2], b_values, blend_factor_new);
}
