#include "../inc/frame_constants.hlsl"
#include "../inc/math.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> csgi_cascade_tex;
[[vk::binding(1)]] cbuffer _ {
    uint cascade_idx;
}

struct VsOut {
	float4 position: SV_Position;
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};

float3 vx_to_pos(float3 vx) {
    vx = csgi_dispatch_vx_to_global_vx(vx, cascade_idx);
    return (vx + 0.5) * csgi_voxel_size(cascade_idx) + CSGI_VOLUME_ORIGIN;
}

VsOut main(uint vid: SV_VertexID, uint instance_index: SV_InstanceID) {
    VsOut vsout;

    const uint face_vert_idx = vid % 6;
    const uint voxel_idx = vid / 6;

    const uint voxel_x = voxel_idx % CSGI_VOLUME_DIMS;
    const uint voxel_y = (voxel_idx / CSGI_VOLUME_DIMS) % CSGI_VOLUME_DIMS;
    const uint voxel_z = (voxel_idx / (CSGI_VOLUME_DIMS * CSGI_VOLUME_DIMS)) % CSGI_VOLUME_DIMS;
    const uint dir_idx = voxel_idx / (CSGI_VOLUME_DIMS * CSGI_VOLUME_DIMS * CSGI_VOLUME_DIMS);
    const uint3 grid_offset = uint3(CSGI_VOLUME_DIMS * dir_idx, 0, 0);

    float4 voxel_packed = csgi_cascade_tex[uint3(voxel_x, voxel_y, voxel_z) + grid_offset];
    if (voxel_packed.a > 0.01) {
        static const float2 face_verts_2d[6] = {
            float2(-1, -1),
            float2(1, 1),
            float2(1, -1),

            float2(1, 1),
            float2(-1, -1),
            float2(-1, 1),
        };

        const float3 slice_dir = CSGI_DIRECT_DIRS[dir_idx];
        const float3x3 slice_rot = build_orthonormal_basis(slice_dir);
        const float3 voxel_center = vx_to_pos(float3(voxel_x, voxel_y, voxel_z));

        const float3 ws_pos = voxel_center + mul(slice_rot, float3(face_verts_2d[face_vert_idx] * 0.5, -0.5) * csgi_voxel_size(cascade_idx));
        const float4 vs_pos = mul(frame_constants.view_constants.world_to_view, float4(ws_pos, 1.0));
        const float4 cs_pos = mul(frame_constants.view_constants.view_to_sample, vs_pos);

        float3 color = voxel_packed.rgb;

        vsout.position = cs_pos;
        //vsout.color = float4(color + 0.01, 1);
        //vsout.color = float4(1.0 - exp(-0.25 * color), 1);
        vsout.color = float4(color, 1);
        vsout.normal = -slice_dir;
    } else {
        vsout.position = 0;
        vsout.color = 0;
        vsout.normal = 0;
    }

    return vsout;
}