#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> direct_tex;
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] Texture3D<float> direct_opacity_tex;
[[vk::binding(3)]] Texture3D<float3> outer_cascade_subray_indirect_tex;
[[vk::binding(4)]] RWTexture3D<float3> subray_indirect_tex;
[[vk::binding(5)]] RWTexture3D<float3> indirect_tex;
[[vk::binding(6)]] cbuffer _ {
    uint cascade_idx;
    uint quantum_idx;
}

static const uint SUBRAY_COUNT = CSGI_CARDINAL_SUBRAY_COUNT;

float4 sample_direct_from(int3 vx, uint dir_idx) {
    if (!gi_volume_contains_vx(frame_constants.gi_cascades[cascade_idx], vx)) {
        return 0.0.xxxx;
    }
    vx = csgi_wrap_vx_within_cascade(vx);

    const int3 offset = int3(CSGI_VOLUME_DIMS * dir_idx, 0, 0);
    float4 val = direct_tex[offset + vx];
    //val.a = val.a * val.a;
    val.rgb /= max(0.01, val.a);
    return max(0.0, val);
}

// TODO: reduce copy-pasta
float3 sample_outer_cascade_subray_indirect_from(int3 vx, uint dir_idx, uint subray, float3 fallback_color) {
    if (!gi_volume_contains_vx(frame_constants.gi_cascades[cascade_idx + 1], vx)) {
        return fallback_color;
    }
    return outer_cascade_subray_indirect_tex[
        csgi_cardinal_vx_dir_subray_to_subray_vx(csgi_wrap_vx_within_cascade(vx), dir_idx, subray)
    ];
}

float3 sample_subray_indirect_from(int3 vx, uint dir_idx, uint subray, float3 fallback_color) {
    // Check if we're out of the cascade, and if so, in which direction.
    // We'll use this to shift the lookup into the parent cascade, favoring
    // sampling towards the inside of the smaller cascade. This avoids leaks
    // when inside buildings, making sure the lookups don't sample the outside.
    const int3 outlier_offset = gi_volume_get_cascade_outlier_offset(frame_constants.gi_cascades[cascade_idx], vx);

    if (any(outlier_offset != 0)) {
        if (cascade_idx < CSGI_CASCADE_COUNT-1) {
            // Note: Only works when the cascade exponential scaling is 2.0; Search token: b518ed19-c715-4cc7-9bc7-e0dbbca3e037
            return sample_outer_cascade_subray_indirect_from(
                // Control rounding, biasing towards the direction of light flow,
                // in order not to jump over small blockers.
                (vx + CSGI_INDIRECT_DIRS[dir_idx]) / 2 - outlier_offset,
                dir_idx,
                subray,
                fallback_color);
        } else {
            return fallback_color;
        }
    }
    return subray_indirect_tex[
        csgi_cardinal_vx_dir_subray_to_subray_vx(csgi_wrap_vx_within_cascade(vx), dir_idx, subray)
    ];
}

void write_subray_indirect_to(float3 radiance, int3 vx, uint dir_idx, uint subray) {
    subray_indirect_tex[
        csgi_cardinal_vx_dir_subray_to_subray_vx(vx, dir_idx, subray)
    ] = prequant_shift_11_11_10(radiance);
}


// TODO: 3D textures on NV seem to be only tiled in the XY plane,
// meaning that the -Z and +Z sweeps are fast, but the others are slow.
// Might want to reshuffle the textures, and then unshuffle them in the "subray combine" pass.
[numthreads(8, 8, 1)]
void main(uint3 dispatch_vx : SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    const uint direct_dir_idx = dispatch_vx.z;
    const uint indirect_dir_idx = direct_dir_idx;
    const int3 slice_dir = CSGI_DIRECT_DIRS[direct_dir_idx];

    float3 atmosphere_color = sky_cube_tex.SampleLevel(sampler_llr, CSGI_INDIRECT_DIRS[indirect_dir_idx].xyz, 0).rgb;

    const int3 indirect_offset = int3(CSGI_VOLUME_DIMS * indirect_dir_idx, 0, 0);

    static const uint TANGENT_COUNT = 4;
    uint tangent_dir_indices[TANGENT_COUNT];
    {for (uint i = 0; i < TANGENT_COUNT; ++i) {
        tangent_dir_indices[i] = ((direct_dir_idx & uint(~1)) + 2 + i) % CSGI_CARDINAL_DIRECTION_COUNT;
    }}

    int slice_z_start = (direct_dir_idx & 1) ? (CSGI_VOLUME_DIMS - 1) : 0; 

    int3 initial_vx;
    if (direct_dir_idx < 2) {
        initial_vx = int3(slice_z_start, dispatch_vx.x, dispatch_vx.y);
    } else if (direct_dir_idx < 4) {
        initial_vx = int3(dispatch_vx.x, slice_z_start, dispatch_vx.y);
    } else {
        initial_vx = int3(dispatch_vx.x, dispatch_vx.y, slice_z_start);
    }

    static const float4 subray_weights[SUBRAY_COUNT] = CSGI_CARDINAL_SUBRAY_TANGENT_WEIGHTS;

    #if 1
    static const uint frame_divisor = 4;
    static const uint plane_start_idx = (frame_constants.frame_index % frame_divisor) * CSGI_VOLUME_DIMS / frame_divisor;
    static const uint plane_end_idx = plane_start_idx + CSGI_VOLUME_DIMS / frame_divisor;
    #else
    static const uint plane_start_idx = 0;
    static const uint plane_end_idx = CSGI_VOLUME_DIMS;
    #endif

    int3 vx = initial_vx - slice_dir * plane_start_idx;
    [loop]
    for (uint slice_z = plane_start_idx; slice_z < plane_end_idx; ++slice_z, vx -= slice_dir) {
        int3 global_vx = csgi_dispatch_vx_to_global_vx(vx, cascade_idx);
        
        const float4 center_direct_s = sample_direct_from(global_vx, direct_dir_idx);
        float4 tangent_neighbor_direct_and_vis[SUBRAY_COUNT];
        float tangent_weights[SUBRAY_COUNT];

        for (uint tangent_i = 0; tangent_i < TANGENT_COUNT; ++tangent_i) {
            const uint tangent_dir_idx = tangent_dir_indices[tangent_i];
            const int3 tangent_dir = CSGI_DIRECT_DIRS[tangent_dir_idx];

            const float center_opacity_t = sample_direct_from(global_vx, tangent_dir_idx).a;
            const float4 direct_neighbor_t = sample_direct_from(global_vx + tangent_dir, tangent_dir_idx);
            const float4 direct_neighbor_s = sample_direct_from(global_vx + tangent_dir, direct_dir_idx);

            float4 neighbor_direct_and_vis = float4(0.0.xxx, 1.0);
            
            neighbor_direct_and_vis = lerp(neighbor_direct_and_vis, float4(direct_neighbor_s.rgb, 0.0), direct_neighbor_s.a);
            neighbor_direct_and_vis = lerp(neighbor_direct_and_vis, float4(direct_neighbor_t.rgb, 0.0), direct_neighbor_t.a);
            neighbor_direct_and_vis = lerp(neighbor_direct_and_vis, 0.0.xxxx, center_opacity_t);
            neighbor_direct_and_vis = lerp(neighbor_direct_and_vis, 0.0.xxxx, center_direct_s.a);

            float wt = 1;

            // TODO: is this right? It does fix the case of bounce on the side of 336_lrm
            wt *= (1 - center_opacity_t);
            wt *= (1 - center_direct_s.a);
            wt *= (1 - 0.75 * direct_neighbor_t.a);

            tangent_neighbor_direct_and_vis[tangent_i] = neighbor_direct_and_vis;
            tangent_weights[tangent_i] = wt;
        }

        float3 subray_radiance[SUBRAY_COUNT] = {
            0.0.xxx, 0.0.xxx, 0.0.xxx, 0.0.xxx, 0.0.xxx,
        };

        {[loop] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
            const float subray_tangent_weights[TANGENT_COUNT] = {
                subray_weights[subray].x,
                subray_weights[subray].y,
                subray_weights[subray].z,
                subray_weights[subray].w,
            };

            float3 scatter = 0.0;
            float scatter_wt = 0.0;

            const float3 center_indirect_s =
                sample_subray_indirect_from(global_vx + slice_dir, direct_dir_idx, subray, atmosphere_color);

            scatter = lerp(center_indirect_s, center_direct_s.rgb, center_direct_s.a);
            scatter_wt += 1;

            for (uint tangent_i = 0; tangent_i < TANGENT_COUNT; ++tangent_i) {
                const uint tangent_dir_idx = tangent_dir_indices[tangent_i];
                const int3 tangent_dir = CSGI_DIRECT_DIRS[tangent_dir_idx];

                float4 neighbor_direct_and_vis = tangent_neighbor_direct_and_vis[tangent_i];
                float3 neighbor_indirect =
                    sample_subray_indirect_from(global_vx + slice_dir + tangent_dir, indirect_dir_idx, subray, atmosphere_color);
                    
                float3 neighbor_radiance = neighbor_direct_and_vis.rgb + neighbor_direct_and_vis.a * neighbor_indirect;
                float wt = subray_tangent_weights[tangent_i] * tangent_weights[tangent_i];

                scatter += neighbor_radiance * wt;
                scatter_wt += wt;
            }

            subray_radiance[subray] += scatter / max(scatter_wt, 1);
        }}

        static const float subray_combine_weights[SUBRAY_COUNT] = {
            1, 1, 1, 1, 1
        };

        float4 combined_indirect = 0.0;
        {[unroll] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
            combined_indirect += float4(subray_radiance[subray], 1) * subray_combine_weights[subray];
            write_subray_indirect_to(subray_radiance[subray], vx, indirect_dir_idx, subray);
        }}

        combined_indirect.rgb /= combined_indirect.a;

        #if CSGI_SUBRAY_COMBINE_DURING_SWEEP
            indirect_tex[vx + int3(CSGI_VOLUME_DIMS * indirect_dir_idx, 0, 0)] = combined_indirect.rgb * direct_opacity_tex[vx];
        #endif
    }
}
