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


[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(1)]] RWStructuredBuffer<float4> surfel_irradiance_buf;
[[vk::binding(2)]] RWStructuredBuffer<float4> surfel_sh_buf;

static const uint MAX_PATH_LENGTH = 5;
static const float3 SUN_DIRECTION = normalize(float3(1, 1.6, -0.2));
static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;

float3 sample_environment_light(float3 dir) {
    //return 0.5.xxx;
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

    float4 total_radiance = 0.0;
    float4 prev_total_radiance_packed = surfel_irradiance_buf[surfel_idx];

    const uint sample_count = clamp(int(32 - prev_total_radiance_packed.w), 1, 32);

    float4 sh_r = 0;
    float4 sh_g = 0;
    float4 sh_b = 0;

    float2 hit_dist_wt = 0;

    for (uint sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        total_radiance.w += 1.0;

        RayDesc outgoing_ray;
        {
            float2 urand = float2(
                uint_to_u01_float(hash1_mut(seed)),
                uint_to_u01_float(hash1_mut(seed))
            );

            /*float3 dir = uniform_sample_sphere(urand);
            if (dot(dir, surfel.normal) < 0) {
                dir = reflect(dir, surfel.normal);
            }*/
            float3 dir = normalize(surfel.normal + uniform_sample_sphere(urand));

            outgoing_ray = new_ray(
                surfel.position,
                normalize(dir),
                1e-3,
                FLT_MAX
            );
        }

        // ----

        float3 throughput = 1.0.xxx;
        float roughness_bias = 0.0;

        //const float surfel_sample_pdf = dot(outgoing_ray.Direction, surfel.normal) / M_PI;
        const float surfel_sample_pdf = 1.0 / M_PI;
        //const float irradiance_contrib = dot(outgoing_ray.Direction, surfel.normal) * M_PI;
        const float irradiance_contrib = 1;
        //const float4 surfel_sample_sh = sh_eval(outgoing_ray.Direction) / max(1e-3, surfel_sample_pdf);
        //float4 surfel_sample_sh = float4(outgoing_ray.Direction, 1);
        float4 surfel_sample_sh = float4(0.4886, outgoing_ray.Direction * 0.282);

        if (dot(outgoing_ray.Direction, surfel.normal) < 0) {
            surfel_sample_sh = 0;
        }

        [loop]
        for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
            const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray);
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

                const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
                const float3 wi = mul(to_light_norm, shading_basis);

                float3 wo = mul(-outgoing_ray.Direction, shading_basis);

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

                const float3 brdf_value = brdf.evaluate(wo, wi);
                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
                const float3 contrib = throughput * brdf_value * light_radiance;

                total_radiance.xyz += contrib * irradiance_contrib;
                sh_r += contrib.r * surfel_sample_sh;
                sh_g += contrib.g * surfel_sample_sh;
                sh_b += contrib.b * surfel_sample_sh;                

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
                    outgoing_ray.Direction = mul(shading_basis, brdf_sample.wi);
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

                total_radiance.xyz += contrib * irradiance_contrib;
                sh_r += contrib.r * surfel_sample_sh;
                sh_g += contrib.g * surfel_sample_sh;
                sh_b += contrib.b * surfel_sample_sh;                

                break;
            }
        }
    }

    total_radiance.xyz /= max(1, total_radiance.w);
    sh_r /= max(1, total_radiance.w);
    sh_g /= max(1, total_radiance.w);
    sh_b /= max(1, total_radiance.w);
    sh_r.yzw /= max(1e-8, sh_r.x);
    sh_g.yzw /= max(1e-8, sh_g.x);
    sh_b.yzw /= max(1e-8, sh_b.x);

    float avg_dist = unpack_dist(hit_dist_wt.x / max(1, hit_dist_wt.y));
    //total_radiance.xyz = lerp(float3(1, 0, 0), float3(0.02, 1.0, 0.2), avg_dist) * total_radiance.w;

    const uint MAX_SAMPLE_COUNT = 64;

    const float total_sample_count = prev_total_radiance_packed.w + total_radiance.w;
    const float blend_factor_new = total_radiance.w / max(1, total_sample_count);
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

    surfel_irradiance_buf[surfel_idx] = float4(lerp(prev_total_radiance_packed.xyz, total_radiance.xyz, blend_factor_new), total_sample_count);
    //surfel_irradiance_buf[surfel_idx] = float4(total_radiance.xyz + 0.5, total_sample_count);
    surfel_sh_buf[surfel_idx * 3 + 0] = lerp(surfel_sh_buf[surfel_idx * 3 + 0], sh_r, blend_factor_new);
    surfel_sh_buf[surfel_idx * 3 + 1] = lerp(surfel_sh_buf[surfel_idx * 3 + 1], sh_g, blend_factor_new);
    surfel_sh_buf[surfel_idx * 3 + 2] = lerp(surfel_sh_buf[surfel_idx * 3 + 2], sh_b, blend_factor_new);
}
