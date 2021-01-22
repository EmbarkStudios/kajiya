#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/bindless_textures.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/mesh.hlsl"


[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> surfel_irradiance_buf;
[[vk::binding(3, 0)]] SamplerState sampler_lnc;

static const uint MAX_PATH_LENGTH = 5;
static const float3 SUN_DIRECTION = normalize(float3(1, 1.6, -0.2));
static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;
static const bool FURNACE_TEST = false;
static const bool FURNACE_TEST_EXCLUDE_DIFFUSE = false;
static const bool USE_PIXEL_FILTER = true;

float3 sample_environment_light(float3 dir) {
    //return 0.5.xxx;

    if (FURNACE_TEST) {
        return 0.5.xxx;
    }

    return atmosphere_default(dir, SUN_DIRECTION);

    float3 col = (dir.zyx * float3(1, 1, -1) * 0.5 + float3(0.6, 0.5, 0.5)) * 0.75;
    col = lerp(col, 1.3.xxx * calculate_luma(col), smoothstep(-0.2, 1.0, dir.y).xxx);
    return col;
}


struct SpecularBrdfEnergyPreservation {
    float3 reflection_mult;
    float3 transmission_fraction;

    static SpecularBrdfEnergyPreservation from_brdf_ndotv(SpecularBrdf brdf, float ndotv) {
        const float roughness = brdf.roughness;
        const float3 specular_albedo = brdf.albedo;

        float2 uv = float2(ndotv, roughness) * BRDF_FG_LUT_UV_SCALE + BRDF_FG_LUT_UV_BIAS;
        float2 fg = bindless_textures[0].SampleLevel(sampler_lnc, uv, 0).xy;

        float3 single_scatter = specular_albedo * fg.x + fg.y;
        float energy_loss_per_bounce = 1.0 - (fg.x + fg.y);
        float3 bounce_radiance = energy_loss_per_bounce * specular_albedo;
        float3 albedo_inf_series = energy_loss_per_bounce * single_scatter / (1.0 - bounce_radiance);
        float3 corrected = single_scatter + albedo_inf_series;

        SpecularBrdfEnergyPreservation res;
        res.reflection_mult = corrected / max(1e-5, single_scatter);
        res.transmission_fraction = 1 - corrected;
        return res;
    }
};

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

    RayDesc outgoing_ray;
    {
        float2 urand = float2(
            uint_to_u01_float(hash1_mut(seed)),
            uint_to_u01_float(hash1_mut(seed))
        );

        outgoing_ray = new_ray(
            surfel.position,
            normalize(surfel.normal + uniform_sample_sphere(urand)),
            0.1,
            FLT_MAX
        );
    }

    // ----

    float3 throughput = 1.0.xxx;
    float3 total_radiance = 0.0.xxx;

    float roughness_bias = 0.0;

    [loop]
    for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
        const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray);
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

            GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
            if (FURNACE_TEST && !FURNACE_TEST_EXCLUDE_DIFFUSE) {
                gbuffer.albedo = 1.0;
            }
            //gbuffer.roughness = lerp(gbuffer.roughness, 1.0, 0.5);
            //gbuffer.metalness = 1.0;
            //gbuffer.albedo = max(gbuffer.albedo, 1e-3);
            //gbuffer.albedo = float3(1, 0.765557, 0.336057);
            //gbuffer.roughness = 0.001;
            //gbuffer.roughness = clamp((int(primary_hit.position.x * 0.2) % 5) / 5.0, 1e-4, 1.0);

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

            SpecularBrdf specular_brdf;            
            specular_brdf.albedo = lerp(0.04, gbuffer.albedo, gbuffer.metalness);

            if (FIREFLY_SUPPRESSION) {
                specular_brdf.roughness = lerp(gbuffer.roughness, 1.0, roughness_bias);
            } else {
                specular_brdf.roughness = gbuffer.roughness;
            }

            DiffuseBrdf diffuse_brdf;
            diffuse_brdf.albedo = max(0.0, 1.0 - gbuffer.metalness) * gbuffer.albedo;

            const float3 albedo_boost = metalness_albedo_boost(gbuffer.metalness, gbuffer.albedo);
            specular_brdf.albedo = min(1.0, specular_brdf.albedo * albedo_boost);
            diffuse_brdf.albedo = min(1.0, diffuse_brdf.albedo * albedo_boost);

            if (FURNACE_TEST && FURNACE_TEST_EXCLUDE_DIFFUSE) {
                diffuse_brdf.albedo = 0.0.xxx;
            }
            
            const SpecularBrdfEnergyPreservation energy_preservation =
                SpecularBrdfEnergyPreservation::from_brdf_ndotv(specular_brdf, wo.z);

            /*const float3 spec_energy_preservation =
                preintegrated_specular_brdf_energy_preservation_mult(specular_brdf.albedo, specular_brdf.roughness, wo.z);*/

            const BrdfValue spec = specular_brdf.evaluate(wo, wi);
            const BrdfValue diff = diffuse_brdf.evaluate(wo, wi);

            //if (path_length > 0)
            if (!FURNACE_TEST) {
                const float3 radiance = (
                    spec.value() * energy_preservation.reflection_mult +
                    diff.value() * spec.transmission_fraction
                ) * max(0.0, wi.z);
                
                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;

                total_radiance += throughput * radiance * light_radiance;
            }

            BrdfSample brdf_sample;
            float lobe_pdf;

            {
                // Sample top level (specular)
                const float u0 = uint_to_u01_float(hash1_mut(seed));
                const float u1 = uint_to_u01_float(hash1_mut(seed));
                brdf_sample = specular_brdf.sample(wo, float2(u0, u1));

                // We should transmit with throughput equal to `brdf_sample.transmission_fraction`,
                // and reflect with the complement of that. However since we use a single ray,
                // we toss a coin, and choose between reflection and transmission.
                const float spec_wt = calculate_luma(brdf_sample.value_over_pdf);
                const float diffuse_wt = calculate_luma(diffuse_brdf.albedo);
                const float transmission_p = diffuse_wt / (spec_wt + diffuse_wt);

                const float lobe_xi = uint_to_u01_float(hash1_mut(seed));
                if (lobe_xi < transmission_p) {
                    // Transmission wins! Now sample the bottom layer (diffuse)

                    lobe_pdf = transmission_p;
                    roughness_bias = lerp(roughness_bias, 1.0, 0.5);

                    // Now account for the masking that the top level exerts on the bottom.
                    // Even though we used `brdf_sample.transmission_fraction` in lobe selection,
                    // that factor cancelled out with division by the lobe selection PDF.
                    //throughput *= brdf_sample.transmission_fraction;
                    //throughput *= 1 - (1 - brdf_sample.transmission_fraction) * spec_energy_preservation;

                    const float u0 = uint_to_u01_float(hash1_mut(seed));
                    const float u1 = uint_to_u01_float(hash1_mut(seed));

                    brdf_sample = diffuse_brdf.sample(wo, float2(u0, u1));
                    throughput *= energy_preservation.transmission_fraction;

                    //throughput *= spec_energy_preservation;
                } else {
                    // Reflection wins!

                    lobe_pdf = (1.0 - transmission_p);
                    roughness_bias = lerp(roughness_bias, 1.0, gbuffer.roughness * 0.5);
                    throughput *= energy_preservation.reflection_mult;
                }
            }

            if (lobe_pdf > 1e-9 && brdf_sample.is_valid()) {
                outgoing_ray.Origin = primary_hit.position;
                outgoing_ray.Direction = mul(shading_basis, brdf_sample.wi);
                outgoing_ray.TMin = 1e-4;
                throughput *= brdf_sample.value_over_pdf / lobe_pdf;
            } else {
                break;
            }

            if (FURNACE_TEST) {
                total_radiance += throughput * sample_environment_light(outgoing_ray.Direction);
                break;
            }
        } else {
            total_radiance += throughput * sample_environment_light(outgoing_ray.Direction);
            break;
        }
    }

    // ----

    float3 prev_total_radiance = surfel_irradiance_buf[surfel_idx].xyz;
    float3 blended_radiance = lerp(prev_total_radiance, total_radiance, 0.01);
    surfel_irradiance_buf[surfel_idx] = float4(blended_radiance, 0);
}
