#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
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
#include "../inc/color.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] StructuredBuffer<VertexPacked> surf_rcache_spatial_buf;
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] RWByteAddressBuffer surf_rcache_grid_meta_buf;
[[vk::binding(3)]] StructuredBuffer<uint> surf_rcache_life_buf;
[[vk::binding(4)]] RWStructuredBuffer<VertexPacked> surf_rcache_reposition_proposal_buf;
[[vk::binding(5)]] RWStructuredBuffer<uint> surf_rcache_reposition_proposal_count_buf;
DEFINE_WRC_BINDINGS(6)
[[vk::binding(7)]] RWByteAddressBuffer surf_rcache_meta_buf;
[[vk::binding(8)]] RWStructuredBuffer<float4> surf_rcache_irradiance_buf;
[[vk::binding(9)]] RWStructuredBuffer<float4> surf_rcache_aux_buf;
[[vk::binding(10)]] RWStructuredBuffer<uint> surf_rcache_pool_buf;
[[vk::binding(11)]] RWStructuredBuffer<uint> surf_rcache_entry_cell_buf;

#include "../inc/sun.hlsl"
#include "../wrc/lookup.hlsl"

//#define SURFEL_LOOKUP_DONT_KEEP_ALIVE
#include "lookup.hlsl"

#define USE_WORLD_RADIANCE_CACHE 0

// Reduces leaks and spatial artifacts,
// but increases temporal fluctuation.
#define USE_DYNAMIC_TRACE_ORIGIN 0

#define USE_BLEND_RESULT 0

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;
static const bool USE_LIGHTS = true;
static const bool USE_EMISSIVE = true;
static const bool SAMPLE_SURFELS_AT_LAST_VERTEX = true;
static const uint MAX_PATH_LENGTH = 1;
static const uint SAMPLES_PER_FRAME = 8;
static const uint TARGET_SAMPLE_COUNT = 512;
static const float SHORT_ESTIMATOR_SAMPLE_COUNT = 3;
static const bool USE_MSME = true;

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

struct SurfelTraceResult {
    float3 incident_radiance;
    float3 direction;
};

SurfelTraceResult surfel_trace(Vertex surfel, DiffuseBrdf brdf, float3x3 tangent_to_world, uint sequence_idx, uint life) {
    uint rng = hash1(sequence_idx);
    //const float2 urand = r2_sequence(sequence_idx % (TARGET_SAMPLE_COUNT * 64));
    const float2 urand = r2_sequence(sequence_idx % max(256, TARGET_SAMPLE_COUNT));

    RayDesc outgoing_ray;
    #if 0 == SURF_RCACHE_USE_SPHERICAL_HARMONICS
    {
        BrdfSample brdf_sample = brdf.sample(float3(0, 0, 1), urand);

        outgoing_ray = new_ray(
            surfel.position,
            mul(tangent_to_world, brdf_sample.wi),
            0.0,
            FLT_MAX
        );
    }
    #else
        outgoing_ray = new_ray(
            surfel.position,
            uniform_sample_sphere(urand),
            0.0,
            FLT_MAX
        );
    #endif

    // force rays in the direction of the normal (debug)
    //outgoing_ray.Direction = mul(tangent_to_world, float3(0, 0, 1));

    SurfelTraceResult result;
    result.direction = outgoing_ray.Direction;

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
    float roughness_bias = 0.5;

    float3 irradiance_sum = 0;
    float2 hit_dist_wt = 0;

    [loop]
    for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
        const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
            .with_cone(RayCone::from_spread_angle(0.03))
            .with_cull_back_faces(false)
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
            
            const float3 urand = float3(
                uint_to_u01_float(hash1_mut(rng)),
                uint_to_u01_float(hash1_mut(rng)),
                uint_to_u01_float(hash1_mut(rng))
            );

            if (SAMPLE_SURFELS_AT_LAST_VERTEX && path_length + 1 == MAX_PATH_LENGTH) {
                irradiance_sum += lookup_surfel_gi(surfel.position, primary_hit.position, gbuffer.normal, 1 + surfel_life_to_rank(life), rng) * throughput * gbuffer.albedo;
            }

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

    result.incident_radiance = irradiance_sum;
    return result;
}

struct Contribution {
    float4 sh_rgb[3];

    void add_radiance_in_direction(float3 radiance, float3 direction) {
        // https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/gdc2018-precomputedgiobalilluminationinfrostbite.pdf
        // `shEvaluateL1`, plus the `4` factor, with `pi` cancelled out in the evaluation code (BRDF).
        float4 sh = float4(0.282095, direction * 0.488603) * 4;
        sh_rgb[0] += sh * radiance.r;
        sh_rgb[1] += sh * radiance.g;
        sh_rgb[2] += sh * radiance.b;
    }

    void scale(float value) {
        sh_rgb[0] *= value;
        sh_rgb[1] *= value;
        sh_rgb[2] *= value;
    }
};

[shader("raygeneration")]
void main() {
    if (SURF_RCACHE_FREEZE) {
        return;
    }

    const uint total_surfel_count = surf_rcache_meta_buf.Load(SURFEL_META_ENTRY_COUNT);
    const uint surfel_idx = DispatchRaysIndex().x;
    const uint life = surf_rcache_life_buf[surfel_idx];

    if (surfel_idx >= total_surfel_count || !is_surfel_life_valid(life)) {
        return;
    }   

    #if USE_DYNAMIC_TRACE_ORIGIN
        const Vertex surfel = unpack_vertex(surf_rcache_reposition_proposal_buf[surfel_idx]);
    #else
        const Vertex surfel = unpack_vertex(surf_rcache_spatial_buf[surfel_idx]);
    #endif

    DiffuseBrdf brdf;
    const float3x3 tangent_to_world = build_orthonormal_basis(surfel.normal);

    brdf.albedo = 1.0.xxx;

    const bool should_reset = all(0.0 == surf_rcache_irradiance_buf[surfel_idx * SURF_RCACHE_IRRADIANCE_STRIDE]);

    if (should_reset) {
        for (uint i = 0; i < SURF_RCACHE_AUX_STRIDE; ++i) {
            surf_rcache_aux_buf[surfel_idx * SURF_RCACHE_AUX_STRIDE + i] = 0.0.xxxx;
        }
    }

    const uint sample_count = SAMPLES_PER_FRAME;
    float3 irradiance_sum = 0;

    // TODO: consider stratifying within cells
    for (uint sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        const uint sequence_idx = hash1(surfel_idx) + sample_idx + frame_constants.frame_index * sample_count;

        SurfelTraceResult traced = surfel_trace(surfel, brdf, tangent_to_world, sequence_idx, life);
        const float3 new_value = traced.incident_radiance;
        const float new_lum = calculate_luma(new_value);

        const float2 octa_coord = octa_encode(normalize(traced.direction));
        const uint2 octa_quant = min(uint2(octa_coord * SURF_RCACHE_OCTA_DIMS), (SURF_RCACHE_OCTA_DIMS - 1).xx);
        const uint octa_idx = octa_quant.x + octa_quant.y * SURF_RCACHE_OCTA_DIMS;

        const uint output_idx = surfel_idx * SURF_RCACHE_AUX_STRIDE + octa_idx;

        const float4 prev_aux = surf_rcache_aux_buf[output_idx];
        const float2 sample_ex_ex2 = float2(new_lum, new_lum * new_lum);

        const float2 prev_ex_ex2 = should_reset ? sample_ex_ex2 : prev_aux.xy;

        const float2 ex_ex2 = lerp(prev_ex_ex2, sample_ex_ex2, 1.0 / min(SHORT_ESTIMATOR_SAMPLE_COUNT, prev_aux.w + 1));

        const float4 new_aux = float4(
            ex_ex2, 0, min(SHORT_ESTIMATOR_SAMPLE_COUNT, prev_aux.w + 1)
        );

        surf_rcache_aux_buf[output_idx] = new_aux;

        const float lum_variance = max(0.0, ex_ex2.y - ex_ex2.x * ex_ex2.x);
        const float lum_dev = sqrt(lum_variance);
        const float quick_lum_ex = ex_ex2.x;

        const float4 prev_value_and_count = surf_rcache_aux_buf[output_idx + SURF_RCACHE_OCTA_DIMS2];
        const float bucket_sample_count = min(1 + prev_value_and_count.w, 32);

        const float3 prev_value = prev_value_and_count.rgb;
        const float prev_lum = calculate_luma(prev_value);

        const float3 prev_value_ycbcr = rgb_to_ycbcr(prev_value);
        const float num_deviations = 1.0;
        float3 prev_value_clamped = ycbcr_to_rgb(sign(prev_value_ycbcr) * clamp(
            abs(prev_value_ycbcr),
            abs(prev_value_ycbcr) * (quick_lum_ex - lum_dev * num_deviations) / max(1e-10, prev_value_ycbcr.x),
            abs(prev_value_ycbcr) * (quick_lum_ex + lum_dev * num_deviations) / max(1e-10, prev_value_ycbcr.x)
        ));
        //prev_value_clamped = prev_value_clamped.xxx;
        //prev_value_clamped = clamp(prev_lum, quick_lum_ex - lum_dev * num_deviations, quick_lum_ex + lum_dev * num_deviations).xxx;

        const float blend_factor_new = 1.0 / bucket_sample_count;
        
        float3 blended_value = lerp(USE_MSME ? prev_value_clamped : prev_value, new_value, blend_factor_new);
        //const float3 blended_value = lerp(prev_value, new_value, blend_factor_new);

        surf_rcache_aux_buf[output_idx + SURF_RCACHE_OCTA_DIMS2] = float4(blended_value, bucket_sample_count);
        irradiance_sum += blended_value;
    }

    irradiance_sum /= sample_count;

    const uint output_idx = surfel_idx * SURF_RCACHE_IRRADIANCE_STRIDE;

    Contribution contribution_sum = (Contribution)0;
    {
        float valid_samples = 0;

        // TODO: counter distortion
        for (uint octa_idx = 0; octa_idx < SURF_RCACHE_OCTA_DIMS2; ++octa_idx) {
            const float2 octa_coord = (float2(octa_idx % SURF_RCACHE_OCTA_DIMS, octa_idx / SURF_RCACHE_OCTA_DIMS) + 0.5) / SURF_RCACHE_OCTA_DIMS;
            const float3 dir = octa_decode(octa_coord);
            const float4 contrib = surf_rcache_aux_buf[surfel_idx * SURF_RCACHE_AUX_STRIDE + SURF_RCACHE_OCTA_DIMS2 + octa_idx];

            contribution_sum.add_radiance_in_direction(
                contrib.rgb,
                dir
            );

            valid_samples += contrib.w > 0 ? 1.0 : 0.0;
        }

        contribution_sum.scale(1.0 / max(1.0, valid_samples));
    }

    for (uint basis_i = 0; basis_i < SURF_RCACHE_IRRADIANCE_STRIDE; ++basis_i) {
        float prev_sample_count = TARGET_SAMPLE_COUNT;
        const float4 new_value = contribution_sum.sh_rgb[basis_i];
        float4 prev_value = surf_rcache_irradiance_buf[surfel_idx * SURF_RCACHE_IRRADIANCE_STRIDE + basis_i];

        if (should_reset) {
            prev_value = new_value;
        }

        const float total_sample_count = prev_sample_count + sample_count;
        float blend_factor_new = 0.25;
        const float4 blended_value = lerp(prev_value, new_value, blend_factor_new);

        surf_rcache_irradiance_buf[surfel_idx * SURF_RCACHE_IRRADIANCE_STRIDE + basis_i] = blended_value;
    }
}
