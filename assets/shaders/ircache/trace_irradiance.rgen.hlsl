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
#include "../inc/reservoir.hlsl"
#include "../inc/bindless_textures.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/mesh.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../wrc/bindings.hlsl"
#include "../inc/color.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] StructuredBuffer<VertexPacked> ircache_spatial_buf;
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] RWByteAddressBuffer ircache_grid_meta_buf;
[[vk::binding(3)]] StructuredBuffer<uint> ircache_life_buf;
[[vk::binding(4)]] RWStructuredBuffer<VertexPacked> ircache_reposition_proposal_buf;
[[vk::binding(5)]] RWStructuredBuffer<uint> ircache_reposition_proposal_count_buf;
DEFINE_WRC_BINDINGS(6)
[[vk::binding(7)]] RWByteAddressBuffer ircache_meta_buf;
[[vk::binding(8)]] RWStructuredBuffer<float4> ircache_irradiance_buf;
[[vk::binding(9)]] RWStructuredBuffer<float4> ircache_aux_buf;
[[vk::binding(10)]] RWStructuredBuffer<uint> ircache_pool_buf;
[[vk::binding(11)]] StructuredBuffer<uint> ircache_entry_indirection_buf;
[[vk::binding(12)]] RWStructuredBuffer<uint> ircache_entry_cell_buf;

#include "../inc/sun.hlsl"
#include "../wrc/lookup.hlsl"

//#define IRCACHE_LOOKUP_DONT_KEEP_ALIVE

// Sample straight from the `ircache_aux_buf` instead of the SH.
#define IRCACHE_LOOKUP_PRECISE

// HACK: reduces feedback loops due to the spherical traces.
// As a side effect, dims down the result a bit, and increases variance.
// Maybe not needed when using IRCACHE_LOOKUP_PRECISE.
#define USE_SELF_LIGHTING_LIMITER 1

#include "lookup.hlsl"

#define USE_WORLD_RADIANCE_CACHE 0

#define USE_BLEND_RESULT 0

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;
static const bool USE_LIGHTS = true;
static const bool USE_EMISSIVE = true;
static const bool SAMPLE_IRCACHE_AT_LAST_VERTEX = true;
static const uint MAX_PATH_LENGTH = 1;
static const uint SAMPLER_SEQUENCE_LENGTH = 1024;
static const uint BUCKET_SAMPLE_COUNT = 8;
static const float SHORT_ESTIMATOR_SAMPLE_COUNT = 3.0;
static const bool USE_MSME = !true;

float3 sample_environment_light(float3 dir) {
    //return 0.0.xxx;
    return sky_cube_tex.SampleLevel(sampler_llr, dir, 0).rgb;
    /*return atmosphere_default(dir, SUN_DIRECTION);

    float3 col = (dir.zyx * float3(1, 1, -1) * 0.5 + float3(0.6, 0.5, 0.5)) * 0.75;
    col = lerp(col, 1.3.xxx * sRGB_to_luminance(col), smoothstep(-0.2, 1.0, dir.y).xxx);
    return col;*/
}

float pack_dist(float x) {
    return min(1, x);
}

float unpack_dist(float x) {
    return x;
}

struct IrcacheTraceResult {
    float3 incident_radiance;
    float3 direction;
    float3 hit_pos;
};

struct SampleParams {
    uint value;

    static SampleParams from_entry_sample_frame(uint entry_idx, uint sample_idx, uint frame_idx) {
        // Checkerboard
        uint cb = sample_idx * 2u + (frame_idx & 1u);
        cb ^= (cb & 4u) >> 2u;

        SampleParams res;
        res.value = cb | ((frame_idx & 0xffff) << 16u) | ((entry_idx & 0xffff) << 4u);

        return res;
    }

    static SampleParams from_raw(uint raw) {
        SampleParams res;
        res.value = raw;
        return res;
    }

    uint raw() {
        return value;
    }

    uint octa_idx() {
        return value & 0x0f;
    }

    uint2 octa_quant() {
        uint oi = octa_idx();
        return uint2(oi & 3, oi >> 2);
    }

    uint rng() {
        return hash1(value >> 4u);
    }

    float2 octa_uv() {
        const uint2 oq = octa_quant();
        const uint r = rng();
        const float2 urand = r2_sequence(r % SAMPLER_SEQUENCE_LENGTH);
        return (float2(oq) + urand) / 4.0;
    }

    // TODO: tackle distortion
    float3 direction() {
        return octa_decode(octa_uv());
    }
};

IrcacheTraceResult ircache_trace(Vertex entry, DiffuseBrdf brdf, SampleParams sample_params, uint life) {
    const float3x3 tangent_to_world = build_orthonormal_basis(entry.normal);

    uint rng = sample_params.rng();

    RayDesc outgoing_ray = outgoing_ray = new_ray(
        entry.position,
        sample_params.direction(),
        0.0,
        FLT_MAX
    );

    // force rays in the direction of the normal (debug)
    //outgoing_ray.Direction = mul(tangent_to_world, float3(0, 0, 1));

    IrcacheTraceResult result;
    result.direction = outgoing_ray.Direction;

    #if USE_WORLD_RADIANCE_CACHE
        WrcFarField far_field =
            WrcFarFieldQuery::from_ray(outgoing_ray.Origin, outgoing_ray.Direction)
                .with_interpolation_urand(float3(
                    uint_to_u01_float(hash1_mut(rng)),
                    uint_to_u01_float(hash1_mut(rng)),
                    uint_to_u01_float(hash1_mut(rng))
                ))
                .with_query_normal(entry.normal)
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

    for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
        const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
            .with_cone(RayCone::from_spread_angle(0.1))
            .with_cull_back_faces(false)
            .with_path_length(path_length + 1)  // +1 because this is indirect light
            .trace(acceleration_structure);

        if (primary_hit.is_hit) {
            if (0 == path_length) {
                result.hit_pos = primary_hit.position;
            }

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

            if (SAMPLE_IRCACHE_AT_LAST_VERTEX && path_length + 1 == MAX_PATH_LENGTH) {
                irradiance_sum += lookup_irradiance_cache(entry.position, primary_hit.position, gbuffer.normal, 1 + ircache_entry_life_to_rank(life), rng) * throughput * gbuffer.albedo;
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
            if (0 == path_length) {
                result.hit_pos = outgoing_ray.Origin + outgoing_ray.Direction * 1000;
            }

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

[shader("raygeneration")]
void main() {
    if (IRCACHE_FREEZE) {
        return;
    }

    const uint dispatch_idx = DispatchRaysIndex().x;
    const uint entry_idx = ircache_entry_indirection_buf[dispatch_idx / IRCACHE_SAMPLES_PER_FRAME];
    const uint sample_idx = dispatch_idx % IRCACHE_SAMPLES_PER_FRAME;
    const uint life = ircache_life_buf[entry_idx];
    const uint rank = ircache_entry_life_to_rank(life);

    VertexPacked packed_entry = ircache_spatial_buf[entry_idx];
    const Vertex entry = unpack_vertex(packed_entry);

    DiffuseBrdf brdf;
    //const float3x3 tangent_to_world = build_orthonormal_basis(entry.normal);

    brdf.albedo = 1.0.xxx;

    // Allocate fewer samples for further bounces
    #if 0
        const uint sample_count_divisor = 
            rank <= 1
            ? 1
            : 4;
    #else
        const uint sample_count_divisor = 1;
    #endif

    const uint sample_count = IRCACHE_SAMPLES_PER_FRAME / sample_count_divisor;

    uint rng = hash1(hash1(entry_idx) + frame_constants.frame_index);

    const SampleParams sample_params = SampleParams::from_entry_sample_frame(entry_idx, sample_idx, frame_constants.frame_index);

    IrcacheTraceResult traced = ircache_trace(entry, brdf, sample_params, life);

    const float self_lighting_limiter = 
        USE_SELF_LIGHTING_LIMITER
        ? lerp(0.75, 1, smoothstep(-0.1, 0, dot(traced.direction, entry.normal)))
        : 1.0;

    const float3 new_value = traced.incident_radiance * self_lighting_limiter;
    const float new_lum = sRGB_to_luminance(new_value);

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState::create();
    Reservoir1spp reservoir = Reservoir1spp::create();
    reservoir.init_with_stream(new_lum, 1.0, stream_state, sample_params.raw());

    const uint octa_idx = sample_params.octa_idx();
    const uint output_idx = entry_idx * IRCACHE_AUX_STRIDE + octa_idx;

    float4 prev_value_and_count =
        // TODO: not correct unless we process every cell just once
        ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2]
        * float4((frame_constants.pre_exposure_delta).xxx, 1);

    float3 val_sel = new_value;
    bool selected_new = true;

    {
        const uint M_CLAMP = 30;

        Reservoir1spp r = Reservoir1spp::from_raw(ircache_aux_buf[output_idx]);
        if (r.M > 0) {
            r.M = min(r.M, M_CLAMP);

            Vertex prev_entry = unpack_vertex(VertexPacked(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2 * 2]));
            //prev_entry.position = entry.position;

            // Validate the previous sample
            if (true) {
                IrcacheTraceResult prev_traced = ircache_trace(prev_entry, brdf, SampleParams::from_raw(r.payload), life);
                {
                    const float prev_self_lighting_limiter = 
                        USE_SELF_LIGHTING_LIMITER
                        ? lerp(0.75, 1, smoothstep(-0.1, 0, dot(prev_traced.direction, prev_entry.normal)))
                        : 1.0;

                    const float3 a = prev_traced.incident_radiance * prev_self_lighting_limiter;
                    const float3 b = prev_value_and_count.rgb;
                    const float3 dist3 = abs(a - b) / (a + b);
                    const float dist = max(dist3.r, max(dist3.g, dist3.b));
                    const float invalidity = smoothstep(0.1, 0.5, dist);
                    r.M = max(0, min(r.M, exp2(log2(float(M_CLAMP)) * (1.0 - invalidity))));

                    // Update the stored value too.
                    // TODO: Feels like the W might need to be updated too, because we would
                    // have picked this sample with a different probability...
                    prev_value_and_count.rgb = a;
                }
            }

            if (reservoir.update_with_stream(
                r, sRGB_to_luminance(prev_value_and_count.rgb), 1.0,
                stream_state, r.payload, rng
            )) {
                val_sel = prev_value_and_count.rgb;
                selected_new = false;
            }
        }
    }


    reservoir.finish_stream(stream_state);

    ircache_aux_buf[output_idx] = reservoir.as_raw();
    ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2] = float4(val_sel, reservoir.W);

    if (selected_new) {
        ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2 * 2] = packed_entry.data0;
    }
}
