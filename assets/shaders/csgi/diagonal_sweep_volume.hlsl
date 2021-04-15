#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> direct_tex;
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] Texture3D<float> direct_opacity_tex;
[[vk::binding(3)]] RWTexture3D<float3> subray_indirect_tex;
[[vk::binding(4)]] RWTexture3D<float3> indirect_tex;

#define USE_DEEP_OCCLUDE 1

static const uint SUBRAY_COUNT = CSGI_DIAGONAL_SUBRAY_COUNT;

float4 sample_direct_from(int3 vx, uint dir_idx) {
    if (any(vx < 0 || vx >= CSGI_VOLUME_DIMS)) {
        return 0.0.xxxx;
    }

    const int3 offset = int3(CSGI_VOLUME_DIMS * dir_idx, 0, 0);
    float4 val = direct_tex[offset + vx];
    val.rgb /= max(0.01, val.a);
    return max(0.0, val);
}

float4 sample_subray_indirect_from(int3 vx, uint dir_idx, uint subray) {
    if (any(vx < 0 || vx >= CSGI_VOLUME_DIMS)) {
        return 0.0.xxxx;
    }

    const int3 indirect_offset = int3(
        SUBRAY_COUNT * CSGI_VOLUME_DIMS * (dir_idx - CSGI_CARDINAL_SUBRAY_COUNT)
        + CSGI_DIAGONAL_DIRECTION_SUBRAY_OFFSET,
        0,
        0);

    const int3 subray_offset = int3(subray, 0, 0);
    const int3 vx_stride = int3(SUBRAY_COUNT, 1, 1);

    return float4(subray_indirect_tex[indirect_offset + subray_offset + vx * vx_stride], 1);
}

void write_subray_indirect_to(float3 radiance, int3 vx, uint dir_idx, uint subray) {
    const int3 indirect_offset = int3(
        SUBRAY_COUNT * CSGI_VOLUME_DIMS * (dir_idx - CSGI_CARDINAL_SUBRAY_COUNT)
        + CSGI_DIAGONAL_DIRECTION_SUBRAY_OFFSET,
        0,
        0);

    const int3 subray_offset = int3(subray, 0, 0);
    const int3 vx_stride = int3(SUBRAY_COUNT, 1, 1);

    subray_indirect_tex[subray_offset + vx * vx_stride + indirect_offset] = prequant_shift_11_11_10(radiance);
}

// 16 threads in a group seem fastest; cache behavior? Not enough threads to fill the GPU with larger groups?
[numthreads(4, 4, 1)]
void main(uint3 dispatch_vx : SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    const uint indirect_dir_idx = CSGI_CARDINAL_DIRECTION_COUNT + dispatch_vx.z;
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
    static const uint plane_end_idx = PLANE_COUNT - plane_start_idx;
    #else
    static const uint plane_start_idx = 0;
    static const uint plane_end_idx = PLANE_COUNT;
    #endif

    static const uint SUBRAY_COUNT = 3;

    {[loop]
    for (uint plane_idx = plane_start_idx; plane_idx < plane_end_idx; ++plane_idx) {
        // A diagonal cross-section of a 3d grid creates a 2d hexagonal grid.
        // Here, axial coordinates are used to find the cells which belong to each slice.
        // See: https://www.redblobgames.com/grids/hexagons/

        const int sum_to = plane_idx;
        const int extent = CSGI_VOLUME_DIMS;
        const int xmin = max(0, sum_to - (extent-1) * 2);
        const int vx_x = dispatch_vx.x + xmin;
        const int ymin = max(0, sum_to - (extent - 1) - vx_x);
        const int vx_y = dispatch_vx.y + ymin;
        int3 vx = int3(vx_x, vx_y, plane_idx - vx_x - vx_y);

        vx = indirect_dir > 0 ? (CSGI_VOLUME_DIMS - vx - 1) : vx;

        if (all(vx >= 0 && vx < extent)) {
            const float4 center_direct_i = sample_direct_from(vx, dir_i_idx);
            const float4 center_direct_i2 = sample_direct_from(vx + dir_i, dir_i_idx);

            static const float skew = 0.333;
            static const float3 subray_wts[SUBRAY_COUNT] = {
                float3(skew, 1.0, 1.0),
                float3(1.0, skew, 1.0),
                float3(1.0, 1.0, skew),
            };
            static const float subray_wt_sum = dot(subray_wts[0], 1.0.xxx);

            float3 subray_radiance[SUBRAY_COUNT] = {
                0.0.xxx, 0.0.xxx, 0.0.xxx,
            };

            {[loop] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float4 indirect_i = 0;

                // Note: one of those branches is faster than none,
                // but two or three are slower than none :shrug:
                //
                // Also note: this only matters if this loop is [unroll]ed.
                // Same performance weirdness applies to [loop], so it's likely not about
                // this branch at all, but the code structure that the branch below
                // (or [unroll] above) triggers.
                //
                //if (center_direct_i.a < 0.999)
                {
                    indirect_i = sample_subray_indirect_from(vx + dir_i, indirect_dir_idx, subray);
                    indirect_i.rgb = lerp(atmosphere_color, indirect_i.rgb, indirect_i.a);
                }

                float wt = subray_wts[subray].x;
                float3 indirect = indirect_i.rgb;
                #if USE_DEEP_OCCLUDE
                    indirect = lerp(indirect, center_direct_i2.rgb, center_direct_i2.a);
                #endif
                indirect = lerp(indirect, center_direct_i.rgb, center_direct_i.a);
                subray_radiance[subray] += indirect * wt;
            }}

            const float4 center_direct_j = sample_direct_from(vx, dir_j_idx);
            const float4 center_direct_j2 = sample_direct_from(vx + dir_j, dir_j_idx);

            {[unroll] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float4 indirect_j = 0;
                //if (center_direct_j.a < 0.999)
                {
                    indirect_j = sample_subray_indirect_from(vx + dir_j, indirect_dir_idx, subray);
                    indirect_j.rgb = lerp(atmosphere_color, indirect_j.rgb, indirect_j.a);
                }

                float wt = subray_wts[subray].y;
                float3 indirect = indirect_j.rgb;
                #if USE_DEEP_OCCLUDE
                    indirect = lerp(indirect, center_direct_j2.rgb, center_direct_j2.a);
                #endif
                indirect = lerp(indirect, center_direct_j.rgb, center_direct_j.a);
                subray_radiance[subray] += indirect * wt;
            }}

            const float4 center_direct_k = sample_direct_from(vx, dir_k_idx);
            const float4 center_direct_k2 = sample_direct_from(vx + dir_k, dir_k_idx);

            {[unroll] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float4 indirect_k = 0;
                //if (center_direct_k.a < 0.999)
                {
                    indirect_k = sample_subray_indirect_from(vx + dir_k, indirect_dir_idx, subray);
                    indirect_k.rgb = lerp(atmosphere_color, indirect_k.rgb, indirect_k.a);
                }

                float wt = subray_wts[subray].z;
                float3 indirect = indirect_k.rgb;
                #if USE_DEEP_OCCLUDE
                    indirect = lerp(indirect, center_direct_k2.rgb, center_direct_k2.a);
                #endif
                indirect = lerp(indirect, center_direct_k.rgb, center_direct_k.a);
                subray_radiance[subray] += indirect * wt;
            }}

            float3 combined_indirect = 0.0.xxx;

            [unroll] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect = subray_radiance[subray] / subray_wt_sum;
                write_subray_indirect_to(indirect, vx, indirect_dir_idx, subray);
                combined_indirect += indirect;
            }

            #if CSGI_SUBRAY_COMBINE_DURING_SWEEP
                indirect_tex[vx + int3(CSGI_VOLUME_DIMS * indirect_dir_idx, 0, 0)] = combined_indirect * (direct_opacity_tex[vx] / SUBRAY_COUNT);
            #endif
        }
    }}
}
