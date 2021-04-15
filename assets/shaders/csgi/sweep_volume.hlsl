#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> direct_tex;
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] Texture3D<float> direct_opacity_tex;
[[vk::binding(3)]] RWTexture3D<float3> subray_indirect_tex;
[[vk::binding(4)]] RWTexture3D<float3> indirect_tex;

static const uint SUBRAY_COUNT = CSGI_CARDINAL_SUBRAY_COUNT;

float4 sample_direct_from(int3 vx, uint dir_idx) {
    if (any(vx < 0 || vx >= CSGI_VOLUME_DIMS)) {
        return 0.0.xxxx;
    }

    const int3 offset = int3(CSGI_VOLUME_DIMS * dir_idx, 0, 0);
    float4 val = direct_tex[offset + vx];
    //val.a = val.a * val.a;
    val.rgb /= max(0.01, val.a);
    return max(0.0, val);
}

float4 sample_subray_indirect_from(int3 vx, uint dir_idx, uint subray) {
    if (any(vx < 0 || vx >= CSGI_VOLUME_DIMS)) {
        return 0.0.xxxx;
    }

    const int3 indirect_offset = int3(SUBRAY_COUNT * CSGI_VOLUME_DIMS * dir_idx, 0, 0);

    const int3 subray_offset = int3(subray, 0, 0);
    const int3 vx_stride = int3(SUBRAY_COUNT, 1, 1);

    return float4(subray_indirect_tex[indirect_offset + subray_offset + vx * vx_stride], 1);
}

void write_subray_indirect_to(float3 radiance, int3 vx, uint dir_idx, uint subray) {
    const int3 indirect_offset = int3(SUBRAY_COUNT * CSGI_VOLUME_DIMS * dir_idx, 0, 0);

    const int3 subray_offset = int3(subray, 0, 0);
    const int3 vx_stride = int3(SUBRAY_COUNT, 1, 1);

    subray_indirect_tex[subray_offset + vx * vx_stride + indirect_offset] = prequant_shift_11_11_10(radiance);
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

    static const float4 subray_weights[SUBRAY_COUNT] = {
        float4(1.0, 1.0, 1.0, 1.0),
        float4(1.0, 0.15, 0.5, 0.5),
        float4(0.15, 1.0, 0.5, 0.5),
        float4(0.5, 0.5, 1.0, 0.15),
        float4(0.5, 0.5, 0.15, 1.0),
    };

    #if 1
    static const uint plane_start_idx = (frame_constants.frame_index % 4) * CSGI_VOLUME_DIMS / 4;
    static const uint plane_end_idx = plane_start_idx + CSGI_VOLUME_DIMS / 4;
    #else
    static const uint plane_start_idx = 0;
    static const uint plane_end_idx = CSGI_VOLUME_DIMS;
    #endif

    int3 vx = initial_vx - slice_dir * plane_start_idx;
    [loop]
    for (uint slice_z = plane_start_idx; slice_z < plane_end_idx; ++slice_z, vx -= slice_dir) {
        const float4 center_direct_s = sample_direct_from(vx, direct_dir_idx);
        float4 tangent_neighbor_direct_and_vis[SUBRAY_COUNT];
        float tangent_weights[SUBRAY_COUNT];

        for (uint tangent_i = 0; tangent_i < TANGENT_COUNT; ++tangent_i) {
            const uint tangent_dir_idx = tangent_dir_indices[tangent_i];
            const int3 tangent_dir = CSGI_DIRECT_DIRS[tangent_dir_idx];

            const float center_opacity_t = sample_direct_from(vx, tangent_dir_idx).a;
            const float4 direct_neighbor_t = sample_direct_from(vx + tangent_dir, tangent_dir_idx);
            const float4 direct_neighbor_s = sample_direct_from(vx + tangent_dir, direct_dir_idx);

            float4 neighbor_direct_and_vis = float4(0.0.xxx, 1.0);
            
            neighbor_direct_and_vis = lerp(neighbor_direct_and_vis, float4(direct_neighbor_s.rgb, 0.0), direct_neighbor_s.a);
            #if 0
                // HACK: ad-hoc scale for off-axis contributions
                neighbor_direct_and_vis = lerp(neighbor_direct_and_vis, float4(0.25 * direct_neighbor_t.rgb, 0.0), direct_neighbor_t.a);
            #else
                neighbor_direct_and_vis = lerp(neighbor_direct_and_vis, float4(direct_neighbor_t.rgb, 0.0), direct_neighbor_t.a);
            #endif
            neighbor_direct_and_vis = lerp(neighbor_direct_and_vis, 0.0.xxxx, center_opacity_t);
            neighbor_direct_and_vis = lerp(neighbor_direct_and_vis, 0.0.xxxx, center_direct_s.a);

            float wt = 1;

            // TODO: is this right? It does fix the case of bounce on the side of 336_lrm
            wt *= (1 - center_opacity_t);
            wt *= (1 - center_direct_s.a);
            //wt *= (1 - 0.75 * direct_neighbor_t.a);

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
                0 == slice_z
                ? atmosphere_color
                : sample_subray_indirect_from(vx + slice_dir, direct_dir_idx, subray).rgb;

            scatter = lerp(center_indirect_s, center_direct_s.rgb, center_direct_s.a);
            scatter_wt += 1;

            for (uint tangent_i = 0; tangent_i < TANGENT_COUNT; ++tangent_i) {
                const uint tangent_dir_idx = tangent_dir_indices[tangent_i];
                const int3 tangent_dir = CSGI_DIRECT_DIRS[tangent_dir_idx];

                float4 neighbor_direct_and_vis = tangent_neighbor_direct_and_vis[tangent_i];
                float3 neighbor_indirect = 0 == slice_z ? atmosphere_color : sample_subray_indirect_from(vx + slice_dir + tangent_dir, indirect_dir_idx, subray).rgb;
                float3 neighbor_radiance = neighbor_direct_and_vis.rgb + neighbor_direct_and_vis.a * neighbor_indirect;
                float wt = subray_tangent_weights[tangent_i] * tangent_weights[tangent_i];

                scatter += neighbor_radiance * wt;
                scatter_wt += wt;
            }

            subray_radiance[subray] += scatter / max(scatter_wt, 1);
        }}

        float3 combined_indirect = 0.0.xxx;

        {[unroll] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
            combined_indirect += subray_radiance[subray];
            write_subray_indirect_to(subray_radiance[subray], vx, indirect_dir_idx, subray);
        }}

        #if CSGI_SUBRAY_COMBINE_DURING_SWEEP
            indirect_tex[vx + int3(CSGI_VOLUME_DIMS * indirect_dir_idx, 0, 0)] = combined_indirect * (direct_opacity_tex[vx] / SUBRAY_COUNT);
        #endif
    }
}
