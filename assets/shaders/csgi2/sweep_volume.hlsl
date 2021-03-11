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
void main(uint2 vx_2d : SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    //return;

    // rays towards -Z, light towards +Z
    const uint direct_dir_idx = 4;
    const uint indirect_dir_idx = 4;
    const int3 slice_dir = CSGI2_SLICE_DIRS[direct_dir_idx];

    //const uint direct_dir_idx = vx_2d.x / CSGI2_VOLUME_DIMS;

    vx_2d.x %= CSGI2_VOLUME_DIMS;
    const int3 direct_offset = int3(CSGI2_VOLUME_DIMS * direct_dir_idx, 0, 0);
    const int3 indirect_offset = int3(CSGI2_VOLUME_DIMS * indirect_dir_idx, 0, 0);

    float3 atmosphere_color = 0;//atmosphere_default(-CSGI2_SLICE_DIRS[direct_dir_idx].xyz, SUN_DIRECTION);

    static const uint TANGENT_COUNT = 4;
    uint tangent_dir_indices[TANGENT_COUNT] = { 0, 1, 2, 3 };  // -X, +X, -Y, +Y

    {[loop]
    for (uint slice_z = 0; slice_z < CSGI2_VOLUME_DIMS; ++slice_z) {
        uint3 vx = uint3(vx_2d, slice_z);
        float3 scatter = 0.0;
        float scatter_wt = 0.0;

        const float4 center_direct_s = sample_direct_from(vx, direct_dir_idx);

        if (center_direct_s.a == 1) {
            scatter = center_direct_s.rgb;
            scatter_wt = 1;
        } else {
            [unroll]
            for (uint tangent_i = 0; tangent_i < TANGENT_COUNT; ++tangent_i) {
                const uint tangent_dir_idx = tangent_dir_indices[tangent_i];
                const int3 tangent_dir = CSGI2_SLICE_DIRS[tangent_dir_idx];

                const float center_opacity_t = sample_direct_from(vx, tangent_dir_idx).a;
                const float4 direct_neighbor_t = sample_direct_from(vx + tangent_dir, tangent_dir_idx);
                const float4 direct_neighbor_s = sample_direct_from(vx + tangent_dir, direct_dir_idx);

                float3 neighbor_radiance = sample_indirect_from(vx + slice_dir + tangent_dir, indirect_dir_idx).rgb;
                neighbor_radiance = lerp(neighbor_radiance, direct_neighbor_s.rgb, direct_neighbor_s.a);
                neighbor_radiance = lerp(neighbor_radiance, 0.0.xxx, direct_neighbor_t.a);
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
