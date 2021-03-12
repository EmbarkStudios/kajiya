#include "../inc/frame_constants.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> direct_tex;
[[vk::binding(1)]] RWTexture3D<float4> indirect_tex;

#define USE_DEEP_OCCLUDE 1

float4 sample_direct_from(int3 vx, uint dir_idx) {
    if (any(vx < 0 || vx >= CSGI2_VOLUME_DIMS)) {
        return 0.0.xxxx;
    }

    const int3 offset = int3(CSGI2_VOLUME_DIMS * dir_idx, 0, 0);
    return direct_tex[offset + vx];
}

float4 sample_indirect_from(int3 vx, uint dir_idx, uint subray) {
    if (any(vx < 0 || vx >= CSGI2_VOLUME_DIMS)) {
        return 0.0.xxxx;
    }

    const int3 offset = int3(CSGI2_VOLUME_DIMS * dir_idx, 0, 0);
    const int3 subray_offset = int3(0, subray * CSGI2_VOLUME_DIMS, 0);

    return indirect_tex[offset + subray_offset + vx];
}

[numthreads(8, 8, 1)]
void main(uint3 dispatch_vx : SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    const uint indirect_dir_idx = CSGI2_SLICE_COUNT + dispatch_vx.z;
    const int3 indirect_dir = CSGI2_INDIRECT_DIRS[indirect_dir_idx];
    const uint dir_i_idx = 0 + (indirect_dir.x > 0 ? 1 : 0);
    const uint dir_j_idx = 2 + (indirect_dir.y > 0 ? 1 : 0);
    const uint dir_k_idx = 4 + (indirect_dir.z > 0 ? 1 : 0);
    const int3 dir_i = CSGI2_SLICE_DIRS[dir_i_idx];
    const int3 dir_j = CSGI2_SLICE_DIRS[dir_j_idx];
    const int3 dir_k = CSGI2_SLICE_DIRS[dir_k_idx];

#if 1
    static const uint PLANE_COUNT = (CSGI2_VOLUME_DIMS - 1) * 3;

    {[loop]
    for (uint plane_idx = 0; plane_idx < PLANE_COUNT; ++plane_idx) {
        const int sum_to = plane_idx;
        const int extent = CSGI2_VOLUME_DIMS;
        const int xmin = max(0, sum_to - (extent-1) * 2);
        const int vx_x = dispatch_vx.x + xmin;
        const int ymin = max(0, sum_to - (extent - 1) - vx_x);
        const int vx_y = dispatch_vx.y + ymin;
        int3 vx = int3(vx_x, vx_y, plane_idx - vx_x - vx_y);

        vx = indirect_dir > 0 ? (CSGI2_VOLUME_DIMS - vx - 1) : vx;

        if (all(vx >= 0 && vx < extent))
        {
#else
    {[loop]
    for (uint slice_z = 0; slice_z < CSGI2_VOLUME_DIMS; ++slice_z) {
        const int3 vx = int3(dispatch_vx.xy, slice_z); {
#endif
            const float4 center_direct_i = sample_direct_from(vx, dir_i_idx);
            const float4 center_direct_j = sample_direct_from(vx, dir_j_idx);
            const float4 center_direct_k = sample_direct_from(vx, dir_k_idx);

            const float4 center_direct_i2 = sample_direct_from(vx + dir_i, dir_i_idx);
            const float4 center_direct_j2 = sample_direct_from(vx + dir_j, dir_j_idx);
            const float4 center_direct_k2 = sample_direct_from(vx + dir_k, dir_k_idx);

            static const float skew = 0.333;
            static const float3 subray_wts[3] = {
                float3(skew, 1.0, 1.0),
                float3(1.0, skew, 1.0),
                float3(1.0, 1.0, skew),
            };

            // [unroll] for (uint subray = 0; subray < 3; ++subray) {
            { uint subray = frame_constants.frame_index % 3;
                float3 scatter = 0.0;
                float scatter_wt = 0.0;

                const float4 indirect_i = sample_indirect_from(vx + dir_i, indirect_dir_idx, subray);
                const float4 indirect_j = sample_indirect_from(vx + dir_j, indirect_dir_idx, subray);
                const float4 indirect_k = sample_indirect_from(vx + dir_k, indirect_dir_idx, subray);

                {
                    float wt = subray_wts[subray].x;
                    float3 indirect = indirect_i.rgb;
                    #if USE_DEEP_OCCLUDE
                        indirect = lerp(indirect, center_direct_i2.rgb, center_direct_i2.a);
                    #endif
                    indirect = lerp(indirect, center_direct_i.rgb, center_direct_i.a);
                    scatter += indirect * wt;
                    scatter_wt += wt;
                }

                {
                    float wt = subray_wts[subray].y;
                    float3 indirect = indirect_j.rgb;
                    #if USE_DEEP_OCCLUDE
                        indirect = lerp(indirect, center_direct_j2.rgb, center_direct_j2.a);
                    #endif
                    indirect = lerp(indirect, center_direct_j.rgb, center_direct_j.a);
                    scatter += indirect * wt;
                    scatter_wt += wt;
                }

                {
                    float wt = subray_wts[subray].z;
                    float3 indirect = indirect_k.rgb;
                    #if USE_DEEP_OCCLUDE
                        indirect = lerp(indirect, center_direct_k2.rgb, center_direct_k2.a);
                    #endif
                    indirect = lerp(indirect, center_direct_k.rgb, center_direct_k.a);
                    scatter += indirect * wt;
                    scatter_wt += wt;
                }

                float4 radiance = float4(scatter / max(scatter_wt, 1), 1);
                const int3 subray_offset = int3(0, subray * CSGI2_VOLUME_DIMS, 0);
                const int3 indirect_offset = int3(indirect_dir_idx * CSGI2_VOLUME_DIMS, 0, 0);
                indirect_tex[subray_offset + vx + indirect_offset] = radiance;
            }
        }
    }}
}
