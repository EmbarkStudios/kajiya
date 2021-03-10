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
[[vk::binding(0)]] RWTexture3D<float4> csgi_cascade0_out_tex;
[[vk::binding(1)]] cbuffer _ {
    uint SWEEP_VX_COUNT;
}

//float4 CSGI2_SLICE_CENTERS[CSGI2_SLICE_COUNT];

#define USE_RAY_JITTER 0

static const float SKY_DIST = 1e5;

float3 vx_to_pos(float3 vx, float3x3 slice_rot, uint gi_slice) {
    //const float3 volume_center = CSGI2_SLICE_CENTERS[gi_slice].xyz;
    const float3 volume_center = gi_volume_center(slice_rot);

    return mul(slice_rot, vx - (CSGI2_VOLUME_DIMS - 1.0) / 2.0) * CSGI2_VOXEL_SIZE + volume_center;
}


[shader("raygeneration")]
void main() {
    uint3 dispatch_px = DispatchRaysIndex().xyz;

    const uint grid_idx = dispatch_px.x / CSGI2_VOLUME_DIMS;
    dispatch_px.x %= CSGI2_VOLUME_DIMS;

    const float3x3 slice_rot = build_orthonormal_basis(CSGI2_SLICE_DIRS[grid_idx].xyz);
    const float3 slice_dir = mul(slice_rot, float3(0, 0, -1));    

    const int slice_z_start = int((dispatch_px.z + 1) * SWEEP_VX_COUNT) - 1;
    const int slice_z_end = int(dispatch_px.z * SWEEP_VX_COUNT) - 1;    

    uint rng = hash2(uint2(frame_constants.frame_index, grid_idx));
    //uint dir_i = frame_constants.frame_index % CSGI2_NEIGHBOR_DIR_COUNT;

    #if USE_RAY_JITTER
        const float offset_x = uint_to_u01_float(hash1_mut(rng)) - 0.5;
        const float offset_y = uint_to_u01_float(hash1_mut(rng)) - 0.5;
        const float blend_factor = 1.0;
    #else
        const float offset_x = 0.0;
        const float offset_y = 0.0;
        const float blend_factor = 1.0;
    #endif

    //uint rng = hash_combine2(hash3(px), frame_constants.frame_index);
    //uint rng = hash1(frame_constants.frame_index);
    //uint rng = hash2(uint2(px.y, frame_constants.frame_index));

    const int3 output_offset = int3(CSGI2_VOLUME_DIMS * grid_idx, 0, 0);
    const int3 pxdir = int3(0, 0, -1);

    int3 px = int3(dispatch_px.xy, slice_z_start);
    while (true) {
        //uint rng = hash2(uint2(dir_i, frame_constants.frame_index));

        const float3 trace_origin = vx_to_pos(px + float3(offset_x, offset_y, 0.5), slice_rot, grid_idx);
        const float3 neighbor_pos = vx_to_pos(px + pxdir + float3(offset_x, offset_y, 0.5), slice_rot, grid_idx);

        const int ray_length_int = px.z - slice_z_end;

        RayDesc outgoing_ray = new_ray(
            trace_origin,
            neighbor_pos - trace_origin,
            0,
            ray_length_int
        );

        //const GbufferPathVertex primary_hit = rt_trace_gbuffer_nocull(acceleration_structure, outgoing_ray, 1.0);

        // TODO: cone spread angle (or use a different rchit shader without cones)
        const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray, 1e2);
        if (primary_hit.is_hit) {
            float3 total_radiance = 0.0.xxx;
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
                gbuffer.roughness = 1.0;

                const float3 gbuffer_normal = primary_hit.gbuffer_packed.unpack_normal();

                // Compensate for lost specular (TODO)
                // Especially important for dark dielectrics which don't bounce light in a diffuse way,
                // but can reflect quite a lot through the specular path.
                // TODO: don't lose saturation.
                const float3 bounce_albedo = lerp(primary_hit.gbuffer_packed.unpack_albedo(), 1.0.xxx, 0.08);
                //const float3 bounce_albedo = primary_hit.gbuffer_packed.unpack_albedo();
                //const float3 bounce_albedo = primary_hit.gbuffer_packed.unpack_albedo();
                
                // Remove contributions where the normal is facing away from
                // the cone direction. This can happen e.g. on rays travelling near floors,
                // where the neighbor rays hit the floors, but contribution should be zero.
                //const float normal_cutoff = smoothstep(0.0, 0.1, dot(slice_dir, -gbuffer_normal));
                //const float normal_cutoff = step(0.0, dot(slice_dir, -gbuffer_normal));
                const float normal_cutoff = dot(slice_dir, -gbuffer_normal);

                if (normal_cutoff > 1e-3) {
                    const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;

    #if 0
                    const float3x3 shading_basis = build_orthonormal_basis(gbuffer_normal);
                    const float3 wi = mul(to_light_norm, shading_basis);
                    float3 wo = normalize(mul(-outgoing_ray.Direction, shading_basis));

                    LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);
                    const float3 brdf_value = brdf.evaluate(wo, wi) * max(0.0, wi.z);
                    total_radiance += brdf_value * light_radiance;
    #else
                    total_radiance +=
                        light_radiance * bounce_albedo * max(0.0, dot(gbuffer_normal, to_light_norm)) / M_PI;
    #endif

                    total_radiance = gbuffer.albedo;

                    #if 0
                        const float3 pos_ws = primary_hit.position;
                        float4 pos_vs = mul(frame_constants.view_constants.world_to_view, float4(pos_ws, 1));
                        const float view_dot = -normalize(pos_vs.xyz).z;

                        float3 v_ws = normalize(mul(frame_constants.view_constants.view_to_world, float4(0, 0, -1, 0)).xyz);

                        total_radiance +=
                            100 * smoothstep(0.997, 1.0, view_dot) * bounce_albedo * max(0.0, dot(gbuffer_normal, -v_ws)) / M_PI;
                    #endif
                }
            }

            int cells_skipped_by_ray = int(floor(primary_hit.ray_t));
            while (cells_skipped_by_ray-- > 0) {
                csgi_cascade0_out_tex[px + output_offset] = lerp(csgi_cascade0_out_tex[px + output_offset], float4(0, 0, 0, 1), blend_factor);
                px += pxdir;
            }

            csgi_cascade0_out_tex[px + output_offset] = lerp(csgi_cascade0_out_tex[px + output_offset], float4(total_radiance, 0), blend_factor);

            px += pxdir;

            if (px.z <= slice_z_end) {
                break;
            }
        } else {
            for (int i = 0; i < ray_length_int; ++i) {
                csgi_cascade0_out_tex[px + output_offset] = lerp(csgi_cascade0_out_tex[px + output_offset], float4(0, 0, 0, 1), blend_factor);
                px += pxdir;
            }

            if (px.z <= slice_z_end) {
                break;
            }
        }
    }
}
