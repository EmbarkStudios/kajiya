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


[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> surfel_irradiance_buf;

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
	return float3(sn * xy, cs * xy, z);
}

[shader("raygeneration")]
void main() {
    const uint surfel_idx = DispatchRaysIndex().x;
    uint seed = hash_combine2(hash1(surfel_idx), frame_constants.frame_index);

    const Vertex surfel = unpack_vertex(surfel_spatial_buf[surfel_idx]);

    float4 total_radiance = 0.0;
    float4 prev_total_radiance_packed = surfel_irradiance_buf[surfel_idx];

    const uint sample_count = clamp(int(32 - prev_total_radiance_packed.w), 1, 32);

    for (uint sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        total_radiance.w += 1.0;

        RayDesc outgoing_ray;
        {
            float2 urand = float2(
                uint_to_u01_float(hash1_mut(seed)),
                uint_to_u01_float(hash1_mut(seed))
            );

            outgoing_ray = new_ray(
                surfel.position,
                normalize(surfel.normal + uniform_sample_sphere(urand)),
                1e-3,
                FLT_MAX
            );
        }

        // ----

        float3 throughput = 1.0.xxx;
        float roughness_bias = 0.0;

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
                total_radiance.xyz += throughput * brdf_value * light_radiance;

                const float3 urand = float3(
                    uint_to_u01_float(hash1_mut(seed)),
                    uint_to_u01_float(hash1_mut(seed)),
                    uint_to_u01_float(hash1_mut(seed))
                );
                BrdfSample brdf_sample = brdf.sample(wo, urand);

                if (brdf_sample.is_valid()) {
                    roughness_bias = lerp(roughness_bias, 1.0, 0.5 * brdf_sample.approx_roughness);
                    outgoing_ray.Origin = primary_hit.position;
                    outgoing_ray.Direction = mul(shading_basis, brdf_sample.wi);
                    outgoing_ray.TMin = 1e-4;
                    throughput *= brdf_sample.value_over_pdf;
                } else {
                    break;
                }
            } else {
                total_radiance.xyz += throughput * sample_environment_light(outgoing_ray.Direction);
                break;
            }
        }
    }

    surfel_irradiance_buf[surfel_idx] = prev_total_radiance_packed + total_radiance;
}
