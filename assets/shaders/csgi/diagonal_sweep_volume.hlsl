#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/math.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> direct_tex;
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] Texture3D<float> direct_opacity_tex;
[[vk::binding(3)]] RWTexture3D<float3> subray_indirect_tex;
[[vk::binding(4)]] RWTexture3D<float3> indirect_tex;

#define USE_DEEP_OCCLUDE 1

static const uint SUBRAY_COUNT = CSGI_DIAGONAL_SUBRAY_COUNT;

bool is_vx_inside_volume(int3 vx) {
    return all(vx >= 0 && vx < CSGI_VOLUME_DIMS);
}

float4 sample_direct_from(int3 vx, uint dir_idx) {
    const int3 offset = int3(CSGI_VOLUME_DIMS * dir_idx, 0, 0);
    float4 val = direct_tex[offset + vx];
    val.rgb /= max(0.01, val.a);
    return max(0.0, val);
}

float3 sample_subray_indirect_from(int3 vx, uint dir_idx, uint subray) {
    const int3 indirect_offset = int3(
        SUBRAY_COUNT * CSGI_VOLUME_DIMS * (dir_idx - CSGI_CARDINAL_DIRECTION_COUNT)
        + CSGI_DIAGONAL_DIRECTION_SUBRAY_OFFSET,
        0,
        0);

    const int3 subray_offset = int3(subray, 0, 0);
    const int3 vx_stride = int3(SUBRAY_COUNT, 1, 1);

    return subray_indirect_tex[indirect_offset + subray_offset + vx * vx_stride];
}

void write_subray_indirect_to(float3 radiance, int3 vx, uint dir_idx, uint subray) {
    const int3 indirect_offset = int3(
        SUBRAY_COUNT * CSGI_VOLUME_DIMS * (dir_idx - CSGI_CARDINAL_DIRECTION_COUNT)
        + CSGI_DIAGONAL_DIRECTION_SUBRAY_OFFSET,
        0,
        0);

    const int3 subray_offset = int3(subray, 0, 0);
    const int3 vx_stride = int3(SUBRAY_COUNT, 1, 1);

    subray_indirect_tex[subray_offset + vx * vx_stride + indirect_offset] = prequant_shift_11_11_10(radiance);
}

void diagonal_sweep_volume(const uint2 dispatch_vx, const uint indirect_dir_idx) {
    const int3 indirect_dir = CSGI_INDIRECT_DIRS[indirect_dir_idx];
    const uint dir_i_idx = 0 + (indirect_dir.x > 0 ? 1 : 0);
    const uint dir_j_idx = 2 + (indirect_dir.y > 0 ? 1 : 0);
    const uint dir_k_idx = 4 + (indirect_dir.z > 0 ? 1 : 0);
    const int3 dir_i = CSGI_DIRECT_DIRS[dir_i_idx];
    const int3 dir_j = CSGI_DIRECT_DIRS[dir_j_idx];
    const int3 dir_k = CSGI_DIRECT_DIRS[dir_k_idx];

    float3 atmosphere_color = sky_cube_tex.SampleLevel(sampler_llr, CSGI_INDIRECT_DIRS[indirect_dir_idx].xyz, 0).rgb;

    static const uint PLANE_COUNT = CSGI_VOLUME_DIMS * 3 - 2;

    #if 1
    static const uint plane_start_idx = (frame_constants.frame_index % 2) * PLANE_COUNT / 2;
    static const uint plane_end_idx = (frame_constants.frame_index % 2 + 1) * PLANE_COUNT / 2;
    #else
    static const uint plane_start_idx = 0;
    static const uint plane_end_idx = PLANE_COUNT;
    #endif

    {[loop]
    for (uint plane_idx = plane_start_idx; plane_idx < plane_end_idx; ++plane_idx) {
        // A diagonal cross-section of a 3d grid creates a 2d hexagonal grid.
        // Here, axial coordinates are used to find the cells which belong to each slice.
        // See: https://www.redblobgames.com/grids/hexagons/

        const int sum_to = plane_idx;
        const int extent = CSGI_VOLUME_DIMS;
        const int xmin = max(0, sum_to - (extent - 1) * 2);
        const int vx_x = dispatch_vx.x + xmin;
        const int ymin = max(0, sum_to - (extent - 1) - vx_x);
        const int vx_y = dispatch_vx.y + ymin;
        int3 vx = int3(vx_x, vx_y, plane_idx - vx_x - vx_y);

        vx = indirect_dir > 0 ? (CSGI_VOLUME_DIMS - vx - 1) : vx;

        [branch]
        if (is_vx_inside_volume(vx)) {
            float4 center_direct_i = sample_direct_from(vx, dir_i_idx);
            float4 center_direct_j = sample_direct_from(vx, dir_j_idx);
            float4 center_direct_k = sample_direct_from(vx, dir_k_idx);

            const bool vx_i_in = is_vx_inside_volume(vx + dir_i);
            const bool vx_j_in = is_vx_inside_volume(vx + dir_j);
            const bool vx_k_in = is_vx_inside_volume(vx + dir_k);

            #if USE_DEEP_OCCLUDE
            {
                const float4 center_direct_i2 = vx_i_in ? sample_direct_from(vx + dir_i, dir_i_idx) : 0.0.xxxx;
                const float4 center_direct_j2 = vx_j_in ? sample_direct_from(vx + dir_j, dir_j_idx) : 0.0.xxxx;
                const float4 center_direct_k2 = vx_k_in ? sample_direct_from(vx + dir_k, dir_k_idx) : 0.0.xxxx;

                center_direct_i = prelerp(center_direct_i, center_direct_i2);
                center_direct_j = prelerp(center_direct_j, center_direct_j2);
                center_direct_k = prelerp(center_direct_k, center_direct_k2);
            }
            #endif

            static const float3 subray_wts[SUBRAY_COUNT] = CSGI_DIAGONAL_SUBRAY_TANGENT_WEIGHTS;
            static const float subray_wt_sum = dot(subray_wts[0], 1.0.xxx);

            float3 subray_radiance[SUBRAY_COUNT] = {
                0.0.xxx, 0.0.xxx, 0.0.xxx,
            };

// Equivalent; speed varies depending on optimizer mood
#if 0
            {for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect = vx_i_in ? sample_subray_indirect_from(vx + dir_i, indirect_dir_idx, subray) : atmosphere_color;

                float wt = subray_wts[subray].x;
                indirect = lerp(indirect, center_direct_i.rgb, center_direct_i.a);
                subray_radiance[subray] += indirect * wt / subray_wt_sum;
            }}

            {for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect = vx_j_in ? sample_subray_indirect_from(vx + dir_j, indirect_dir_idx, subray) : atmosphere_color;

                float wt = subray_wts[subray].y;
                indirect = lerp(indirect, center_direct_j.rgb, center_direct_j.a);
                subray_radiance[subray] += indirect * wt / subray_wt_sum;
            }}

            {for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect = vx_k_in ? sample_subray_indirect_from(vx + dir_k, indirect_dir_idx, subray) : atmosphere_color;

                float wt = subray_wts[subray].z;
                indirect = lerp(indirect, center_direct_k.rgb, center_direct_k.a);
                subray_radiance[subray] += indirect * wt / subray_wt_sum;
            }}
#else
            {[unroll] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect_i = vx_i_in ? sample_subray_indirect_from(vx + dir_i, indirect_dir_idx, subray) : atmosphere_color;
                float3 indirect_j = vx_j_in ? sample_subray_indirect_from(vx + dir_j, indirect_dir_idx, subray) : atmosphere_color;
                float3 indirect_k = vx_k_in ? sample_subray_indirect_from(vx + dir_k, indirect_dir_idx, subray) : atmosphere_color;

                indirect_i = lerp(indirect_i, center_direct_i.rgb, center_direct_i.a);
                indirect_j = lerp(indirect_j, center_direct_j.rgb, center_direct_j.a);
                indirect_k = lerp(indirect_k, center_direct_k.rgb, center_direct_k.a);

                const float3 wt = subray_wts[subray];
                subray_radiance[subray] = (indirect_i * wt.x + indirect_j * wt.y + indirect_k * wt.z) / subray_wt_sum;
            }}
#endif

            float3 combined_indirect = 0.0.xxx;

            {[unroll] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect = subray_radiance[subray];
                write_subray_indirect_to(indirect, vx, indirect_dir_idx, subray);
                combined_indirect += indirect;
            }}

            #if CSGI_SUBRAY_COMBINE_DURING_SWEEP
                indirect_tex[vx + int3(CSGI_VOLUME_DIMS * indirect_dir_idx, 0, 0)]
                    = combined_indirect * (direct_opacity_tex[vx] / SUBRAY_COUNT);
            #endif
        }
    }}
}

// 16 threads in a group seem fastest; cache behavior? Not enough threads to fill the GPU with larger groups?
[numthreads(4, 4, 1)]
void main(uint3 dispatch_vx : SV_DispatchThreadID) {
    #if 1
        switch (dispatch_vx.z) {
            case 0: diagonal_sweep_volume(dispatch_vx.xy, CSGI_CARDINAL_DIRECTION_COUNT + 0); break;
            case 1: diagonal_sweep_volume(dispatch_vx.xy, CSGI_CARDINAL_DIRECTION_COUNT + 1); break;
            case 2: diagonal_sweep_volume(dispatch_vx.xy, CSGI_CARDINAL_DIRECTION_COUNT + 2); break;
            case 3: diagonal_sweep_volume(dispatch_vx.xy, CSGI_CARDINAL_DIRECTION_COUNT + 3); break;
            case 4: diagonal_sweep_volume(dispatch_vx.xy, CSGI_CARDINAL_DIRECTION_COUNT + 4); break;
            case 5: diagonal_sweep_volume(dispatch_vx.xy, CSGI_CARDINAL_DIRECTION_COUNT + 5); break;
            case 6: diagonal_sweep_volume(dispatch_vx.xy, CSGI_CARDINAL_DIRECTION_COUNT + 6); break;
            case 7: diagonal_sweep_volume(dispatch_vx.xy, CSGI_CARDINAL_DIRECTION_COUNT + 7); break;
        }
    #else
        const uint indirect_dir_idx = CSGI_CARDINAL_DIRECTION_COUNT + dispatch_vx.z;
        diagonal_sweep_volume(dispatch_vx.xy, indirect_dir_idx);
    #endif
}
