#include "../inc/samplers.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"

#include "common.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0)]] Texture3D<float4> csgi_indirect_tex;
[[vk::binding(1)]] RWTexture3D<float4> csgi_direct_tex;
[[vk::binding(2)]] cbuffer _ {
    uint SWEEP_VX_COUNT;
}

#include "lookup.hlsl"

//float4 CSGI_SLICE_CENTERS[CSGI_CARDINAL_DIRECTION_COUNT];

// TODO: maybe trace multiple rays per frame instead. The delay is not awesome.
// Or use a more advanced temporal integrator, e.g. variance-aware exponential smoothing
#define USE_RAY_JITTER 1
#define RAY_JITTER_AMOUNT 0.75
//#define RAY_JITTER_AMOUNT 1.0

#define USE_MULTIBOUNCE 1

// HACK; TODO: find all the energy loss
#define MULTIBOUNCE_SCALE 1.0

static const float SKY_DIST = 1e5;

float3 vx_to_pos(float3 vx) {
    const float3 volume_center = CSGI_VOLUME_CENTER;
    return (vx - (CSGI_VOLUME_DIMS - 1.0) / 2.0) * CSGI_VOXEL_SIZE + volume_center;
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

    uint rng = hash1(frame_constants.frame_index);
    //uint dir_i = frame_constants.frame_index % CSGI_NEIGHBOR_DIR_COUNT;

    #if USE_RAY_JITTER
        const float jitter_amount = RAY_JITTER_AMOUNT;
        //const float offset_x = (uint_to_u01_float(hash1_mut(rng)) - 0.5) * jitter_amount;
        //const float offset_y = (uint_to_u01_float(hash1_mut(rng)) - 0.5) * jitter_amount;
        float2 jitter_urand = hammersley(frame_constants.frame_index % 4, 8);
        const float offset_x = jitter_urand.x * jitter_amount;
        const float offset_y = jitter_urand.y * jitter_amount;
        const float blend_factor = CSGI_ACCUM_HYSTERESIS;
    #else
        const float offset_x = 0.0;
        const float offset_y = 0.0;
        const float blend_factor = CSGI_ACCUM_HYSTERESIS;
    #endif

    float3 vx_trace_offset;
    int3 vx;

    // Start at the start of the cell, trace a ray to the end
    const float cell_ray_start_offset = -0.5;

    if (grid_idx < 2) {
        vx_trace_offset = float3(cell_ray_start_offset * slice_dir.x, offset_x, offset_y);
        vx = int3(slice_z_start, dispatch_vx.x, dispatch_vx.y);
    } else if (grid_idx < 4) {
        vx_trace_offset = float3(offset_x, cell_ray_start_offset * slice_dir.y, offset_y);
        vx = int3(dispatch_vx.x, slice_z_start, dispatch_vx.y);
    } else {
        vx_trace_offset = float3(offset_x, offset_y, cell_ray_start_offset * slice_dir.z);
        vx = int3(dispatch_vx.x, dispatch_vx.y, slice_z_start);
    }

    const int3 output_offset = int3(CSGI_VOLUME_DIMS * grid_idx, 0, 0);
    int ray_length_int = SWEEP_VX_COUNT;

    while (ray_length_int > 0) {
        const float3 trace_origin = vx_to_pos(vx + vx_trace_offset);
        const float3 neighbor_pos = vx_to_pos(vx + slice_dir + vx_trace_offset);

        RayDesc outgoing_ray = new_ray(
            lerp(trace_origin, neighbor_pos, 1e-3),
            neighbor_pos - trace_origin,
            0,
            ray_length_int
        );

        //const GbufferPathVertex primary_hit = rt_trace_gbuffer_nocull(acceleration_structure, outgoing_ray, 1.0);

        // TODO: cone spread angle (or use a different rchit shader without cones)
        // Note: rt_trace_gbuffer_nocull _might_ be more watertight (needs research)
        // but it does end up losing a lot of energy near geometric complexity
        const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray, 1e2);
        if (primary_hit.is_hit) {
            float4 total_radiance = 0.0.xxxx;
            {
                const float3 to_light_norm = SUN_DIRECTION;
                const bool is_shadowed =
                    rt_is_shadowed(
                        acceleration_structure,
                        new_ray(
                            primary_hit.position,
                            to_light_norm,
                            1e-4,
                            SKY_DIST
                    ));

                GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
                //gbuffer.roughness = lerp(gbuffer.roughness, 1.0, 0.2);
                gbuffer.roughness = 1;

                const float3 gbuffer_normal = primary_hit.gbuffer_packed.unpack_normal();

                // Compensate for lost specular (TODO)
                // Especially important for dark dielectrics which don't bounce light in a diffuse way,
                // but can reflect quite a lot through the specular path.
                // TODO: don't lose saturation.
                //const float3 bounce_albedo = lerp(primary_hit.gbuffer_packed.unpack_albedo(), 1.0.xxx, 0.08);
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

                    const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;

    #if 0
                    const float3x3 shading_basis = build_orthonormal_basis(gbuffer_normal);
                    const float3 wi = mul(to_light_norm, shading_basis);
                    float3 wo = normalize(mul(-outgoing_ray.Direction, shading_basis));

                    LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);
                    const float3 brdf_value = brdf.evaluate(wo, wi) * max(0.0, wi.z);
                    radiance_contribution += brdf_value * light_radiance;
    #else
                    radiance_contribution +=
                        light_radiance * bounce_albedo * max(0.0, dot(gbuffer_normal, to_light_norm)) / M_PI;
    #endif

                    radiance_contribution += gbuffer.emissive;

                    //radiance_contribution = gbuffer.albedo + 0.1;

                    if (USE_MULTIBOUNCE) {
                        radiance_contribution += lookup_csgi(
                            primary_hit.position,
                            gbuffer_normal,
                            CsgiLookupParams::make_default()
                                .with_linear_fetch(false)
                        ) * bounce_albedo * MULTIBOUNCE_SCALE;;
                    }

                    total_radiance += float4(radiance_contribution, 1.0);
                }
            }

            int cells_skipped_by_ray = int(floor(primary_hit.ray_t));
            ray_length_int -= cells_skipped_by_ray + 1;
            vx += slice_dir * cells_skipped_by_ray;

            csgi_direct_tex[vx + output_offset] = lerp(
                // Cancel the decay that runs just before this pass
                csgi_direct_tex[vx + output_offset] / (1.0 - CSGI_ACCUM_HYSTERESIS),
                //total_radiance,
                float4(total_radiance.xyz, 1.0),
                blend_factor
            );
            vx += slice_dir;
        } else {
            break;
        }
    }
}
