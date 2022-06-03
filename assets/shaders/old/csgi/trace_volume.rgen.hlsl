#include "../inc/samplers.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/lights/triangle.hlsl"

#include "common.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0)]] Texture3D<float4> csgi_indirect_tex[CSGI_CASCADE_COUNT];
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] RWTexture3D<float4> csgi_direct_tex;
[[vk::binding(3)]] cbuffer _ {
    uint SWEEP_VX_COUNT;
    uint cascade_idx;
    uint quantum_idx;
}

#include "lookup.hlsl"

#define USE_SOFT_SHADOWS 0

// TODO: maybe trace multiple rays per frame instead. The delay is not awesome.
// Or use a more advanced temporal integrator, e.g. variance-aware exponential smoothing
#define USE_RAY_JITTER 1

// Shrunk a bit to reduce the chance of going across walls
#define RAY_JITTER_AMOUNT 0.75

// Most correct, most leaky.
// TODO: the "wiggle room" around each cell could be estimate by looking at hit distances
// of perpendicular rays, and finding a conservative bounding box. This would allow wide jittering,
// while avoiding excessive sampling in blocked directions.
//
// Another approach would be to shoot rays from a single point, but in a range of angles,
// aiming for a set of points on the neighboring cell's entry face. This would have the downside
// of preventing the strided ray tracing optimization.
//#define RAY_JITTER_AMOUNT 1.0

#define USE_MULTIBOUNCE 1
#define USE_EMISSIVE 1
#define USE_LIGHTS 1

static const float SKY_DIST = 1e5;

float3 vx_to_pos(int3 vx) {
    vx = csgi_dispatch_vx_to_global_vx(vx, cascade_idx);
    return (vx + 0.5) * csgi_voxel_size(cascade_idx) + CSGI_VOLUME_ORIGIN;
}


[shader("raygeneration")]
void main() {
    uint3 dispatch_vx = DispatchRaysIndex().xyz;

    const uint grid_idx = dispatch_vx.x / CSGI_VOLUME_DIMS;
    dispatch_vx.x %= CSGI_VOLUME_DIMS;

    const int3 slice_dir = CSGI_DIRECT_DIRS[grid_idx]; 

    int slice_z_start = int(dispatch_vx.z * SWEEP_VX_COUNT);

    if (0 == (grid_idx & 1)) {
        // Going in the negative direction of an axis. Start at the high end of the slice.
        slice_z_start += SWEEP_VX_COUNT - 1;
    }

    #if USE_RAY_JITTER
        const float jitter_amount = RAY_JITTER_AMOUNT;
        //const float offset_a = (uint_to_u01_float(hash1_mut(rng)) - 0.5) * jitter_amount;
        //const float offset_b = (uint_to_u01_float(hash1_mut(rng)) - 0.5) * jitter_amount;
        float2 jitter_urand = hammersley(frame_constants.frame_index % 4, 8);
        const float offset_a = jitter_urand.x * jitter_amount;
        const float offset_b = jitter_urand.y * jitter_amount;
    #else
        const float offset_a = 0.0;
        const float offset_b = 0.0;
    #endif

    float3 vx_trace_offset;
    int3 vx;

    // Start at the start of the cell, trace a ray to the end
    const float cell_ray_start_offset = -0.5;

    if (grid_idx < 2) {
        vx_trace_offset = float3(cell_ray_start_offset * slice_dir.x, offset_a, offset_b);
        vx = int3(slice_z_start, dispatch_vx.x, dispatch_vx.y);
    } else if (grid_idx < 4) {
        vx_trace_offset = float3(offset_a, cell_ray_start_offset * slice_dir.y, offset_b);
        vx = int3(dispatch_vx.x, slice_z_start, dispatch_vx.y);
    } else {
        vx_trace_offset = float3(offset_a, offset_b, cell_ray_start_offset * slice_dir.z);
        vx = int3(dispatch_vx.x, dispatch_vx.y, slice_z_start);
    }

    uint rng = hash4(uint4(vx, frame_constants.frame_index));

    const int3 output_offset = int3(CSGI_VOLUME_DIMS * grid_idx, 0, 0);
    int ray_length_int = SWEEP_VX_COUNT;

    while (ray_length_int > 0) {
        const float3 trace_origin = vx_to_pos(vx) + vx_trace_offset * csgi_voxel_size(cascade_idx);
        const float3 neighbor_pos = vx_to_pos(vx) + (slice_dir + vx_trace_offset) * csgi_voxel_size(cascade_idx);

        RayDesc outgoing_ray = new_ray(
            lerp(trace_origin, neighbor_pos, 1e-3),
            neighbor_pos - trace_origin,
            0,
            ray_length_int
        );

        // TODO: cone spread angle (or use a different rchit shader without cones)
        // Note: rt_trace_gbuffer_nocull might be more watertight (needs research)
        // but it does end up losing a lot of energy near geometric complexity
        // Note2: actually without _nocull, the GI can spread through backfaces, causing major leaks in scenes which
        // appear watertight, but are leaky from the outside, e.g. "kitchen-interior"
        const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
            .with_cone(RayCone::from_spread_angle(1.0))
            .with_cull_back_faces(false)
            .with_path_length(1)
            .trace(acceleration_structure);

        if (primary_hit.is_hit) {
            float4 total_radiance = 0.0.xxxx;
            {
                GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
                gbuffer.roughness = 1;

                const float3 gbuffer_normal = primary_hit.gbuffer_packed.unpack_normal();

                // Compensate for lost specular (TODO)
                // Especially important for dark dielectrics which don't bounce light in a diffuse way,
                // but can reflect quite a lot through the specular path.
                // TODO: don't lose saturation.
                const float3 bounce_albedo = lerp(primary_hit.gbuffer_packed.unpack_albedo(), 1.0.xxx, 0.04);
                //const float3 bounce_albedo = primary_hit.gbuffer_packed.unpack_albedo();
                
                // Remove contributions where the normal is facing away from
                // the cone direction. This can happen e.g. on rays travelling near floors,
                // where the neighbor rays hit the floors, but contribution should be zero.
                //const float normal_cutoff = smoothstep(0.0, 0.1, dot(slice_dir, -gbuffer_normal));
                //const float normal_cutoff = step(0.0, dot(slice_dir, -gbuffer_normal));
                const float normal_cutoff = dot(float3(slice_dir), -gbuffer_normal);

                if (normal_cutoff > 1e-3) {
                    float3 radiance_contribution = 0.0.xxx;

                    // Sun
                    {
                        const float3 to_sun_norm = SUN_DIRECTION;
                        const bool is_sun_shadowed =
                            rt_is_shadowed(
                                acceleration_structure,
                                new_ray(
                                    primary_hit.position,
                                    to_sun_norm,
                                    1e-4,
                                    SKY_DIST
                            ));
                        const float3 light_radiance = is_sun_shadowed ? 0.0 : SUN_COLOR;

                        radiance_contribution +=
                            light_radiance * bounce_albedo * max(0.0, dot(gbuffer_normal, to_sun_norm)) / M_PI;
                    }

                    if (USE_LIGHTS) {
                        for (uint light_idx = 0; light_idx < frame_constants.triangle_light_count; light_idx += 1) {
                            const uint light_sample_count = 4;
                            const float2 light_rand_base = float2(
                                uint_to_u01_float(hash1_mut(rng)),
                                uint_to_u01_float(hash1_mut(rng))
                            );

                            for (uint light_sample_i = 0; light_sample_i < light_sample_count; light_sample_i += 1) {
                                const float2 urand = frac(light_rand_base + hammersley(light_sample_i % light_sample_count, light_sample_count));

                                TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
                                LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
                                const float3 shadow_ray_origin = primary_hit.position;
                                const float3 to_light_ws = light_sample.pos - primary_hit.position;
                                const float dist_to_light2 = dot(to_light_ws, to_light_ws);
                                const float3 to_light_norm_ws = to_light_ws * rsqrt(dist_to_light2);

                                const float to_psa_metric =
                                    max(0.0, dot(to_light_norm_ws, gbuffer_normal))
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

                                    radiance_contribution +=
                                        !is_shadowed ?
                                        (
                                            triangle_light.radiance() * bounce_albedo / light_sample.pdf.value * to_psa_metric / M_PI / light_sample_count
                                        ) : 0;
                                }
                            }
                        }
                    }
    
                    #if USE_EMISSIVE
                        radiance_contribution += gbuffer.emissive;
                    #endif

                    //radiance_contribution = gbuffer.albedo + 0.1;

                    if (USE_MULTIBOUNCE) {
                        radiance_contribution += lookup_csgi(
                            primary_hit.position,
                            gbuffer_normal,
                            CsgiLookupParams::make_default()
                                .with_linear_fetch(false)
                        ) * bounce_albedo;
                    }

                    total_radiance += float4(radiance_contribution, 1.0);
                }
            }

            int cells_skipped_by_ray = int(floor(primary_hit.ray_t));
            ray_length_int -= cells_skipped_by_ray + 1;
            vx += slice_dir * cells_skipped_by_ray;

            if (csgi_was_dispatch_vx_just_scrolled_in(vx, cascade_idx)) {
                // Just revealed by volume scrolling. Overwrite instead of blending.
                csgi_direct_tex[vx + output_offset] = float4(total_radiance.xyz, 1.0);
            } else {
                csgi_direct_tex[vx + output_offset] = lerp(
                    // Cancel the decay that runs just before this pass
                    csgi_direct_tex[vx + output_offset] / (1.0 - CSGI_ACCUM_HYSTERESIS),
                    //total_radiance,
                    float4(total_radiance.xyz, 1.0),
                    CSGI_ACCUM_HYSTERESIS
                );
            }

            vx += slice_dir;
        } else {
            break;
        }
    }
}
