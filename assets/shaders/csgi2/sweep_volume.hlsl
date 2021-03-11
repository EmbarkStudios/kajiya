#include "../inc/frame_constants.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> direct_tex;
[[vk::binding(1)]] RWTexture3D<float4> indirect_tex;

float4 sample_direct_from(int3 vx, uint dir_idx) {
    const int3 offset = int3(CSGI2_VOLUME_DIMS * dir_idx, 0, 0);
    return direct_tex[offset + vx];
}

float4 sample_indirect_from(int3 vx, uint dir_idx) {
    const int3 offset = int3(CSGI2_VOLUME_DIMS * dir_idx, 0, 0);
    return indirect_tex[offset + vx];
}

[numthreads(8, 8, 1)]
void main(uint3 dispatch_vx : SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    const uint direct_dir_idx = dispatch_vx.z;
    const uint indirect_dir_idx = direct_dir_idx;
    const int3 slice_dir = CSGI2_SLICE_DIRS[direct_dir_idx];

    const int3 direct_offset = int3(CSGI2_VOLUME_DIMS * direct_dir_idx, 0, 0);
    const int3 indirect_offset = int3(CSGI2_VOLUME_DIMS * indirect_dir_idx, 0, 0);

    static const uint TANGENT_COUNT = 4;
    uint tangent_dir_indices[TANGENT_COUNT];
    {for (uint i = 0; i < TANGENT_COUNT; ++i) {
        tangent_dir_indices[i] = ((direct_dir_idx & uint(~1)) + 2 + i) % CSGI2_SLICE_COUNT;
    }}

    int slice_z_start = (direct_dir_idx & 1) * CSGI2_VOLUME_DIMS;

    int3 vx;
    if (direct_dir_idx < 2) {
        vx = int3(slice_z_start, dispatch_vx.x, dispatch_vx.y);
    } else if (direct_dir_idx < 4) {
        vx = int3(dispatch_vx.x, slice_z_start, dispatch_vx.y);
    } else {
        vx = int3(dispatch_vx.x, dispatch_vx.y, slice_z_start);
    }

    {[loop]
    for (uint slice_z = 0; slice_z < CSGI2_VOLUME_DIMS; ++slice_z, vx -= slice_dir) {
        float3 scatter = 0.0;
        float scatter_wt = 0.0;

        const float4 center_direct_s = sample_direct_from(vx, direct_dir_idx);

        if (center_direct_s.a == 1) {
            scatter = center_direct_s.rgb;
            scatter_wt = 1;
        } else {
            scatter = sample_indirect_from(vx + slice_dir, direct_dir_idx).rgb;
            scatter_wt = 1;

            [unroll]
            for (uint tangent_i = 0; tangent_i < TANGENT_COUNT; ++tangent_i) {
                const uint tangent_dir_idx = tangent_dir_indices[tangent_i];
                const int3 tangent_dir = CSGI2_SLICE_DIRS[tangent_dir_idx];

                const float center_opacity_t = sample_direct_from(vx, tangent_dir_idx).a;
                const float4 direct_neighbor_t = sample_direct_from(vx + tangent_dir, tangent_dir_idx);
                const float4 direct_neighbor_s = sample_direct_from(vx + tangent_dir, direct_dir_idx);

                float3 neighbor_radiance = sample_indirect_from(vx + slice_dir + tangent_dir, indirect_dir_idx).rgb;
                
                neighbor_radiance = lerp(neighbor_radiance, direct_neighbor_s.rgb, direct_neighbor_s.a);
                //neighbor_radiance = lerp(neighbor_radiance, 0.0.xxx, direct_neighbor_t.a);
                neighbor_radiance = lerp(neighbor_radiance, direct_neighbor_t.rgb, direct_neighbor_t.a);
                neighbor_radiance = lerp(neighbor_radiance, 0.0.xxx, center_opacity_t);
                neighbor_radiance = lerp(neighbor_radiance, 0.0.xxx, center_direct_s.a);

                scatter += neighbor_radiance;
                scatter_wt += 1;
            }
        }

        float4 radiance = float4(scatter.xyz / max(scatter_wt, 1), 1);
        indirect_tex[vx + indirect_offset] = radiance;
    }}
}
