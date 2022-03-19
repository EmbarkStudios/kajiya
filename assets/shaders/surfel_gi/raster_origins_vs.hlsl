#include "../inc/frame_constants.hlsl"
#include "../inc/math.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked
#include "surfel_constants.hlsl"

[[vk::binding(0)]] ByteAddressBuffer surf_rcache_meta_buf;
[[vk::binding(1)]] ByteAddressBuffer surf_rcache_grid_meta_buf;
[[vk::binding(2)]] StructuredBuffer<VertexPacked> surf_rcache_spatial_buf;

struct VsOut {
	float4 position: SV_Position;
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};

static const int3 FACE_DIRS[6] = {
    int3(-1, 0, 0),
    int3(1, 0, 0),
    int3(0, -1, 0),
    int3(0, 1, 0),
    int3(0, 0, -1),
    int3(0, 0, 1)
};

VsOut main(uint vid: SV_VertexID, uint instance_index: SV_InstanceID) {
    VsOut vsout;

    const uint face_vert_idx = vid % 6;
    const uint dir_idx = (vid / 6) % 6;
    const uint entry_idx = vid / 6 / 6;

    //const uint entry_flags = surf_rcache_grid_meta_buf.Load(sizeof(uint4) * cell_idx + sizeof(uint));
    const float3 voxel_center = unpack_vertex(surf_rcache_spatial_buf[entry_idx]).position;

    if (voxel_center.x != asfloat(0)) {
        static const float2 face_verts_2d[6] = {
            float2(-1, -1),
            float2(1, 1),
            float2(1, -1),

            float2(1, 1),
            float2(-1, -1),
            float2(-1, 1),
        };

        const float3 slice_dir = FACE_DIRS[dir_idx];
        const float3x3 slice_rot = build_orthonormal_basis(slice_dir);

        const float voxel_size = 0.02;

        const float3 ws_pos = voxel_center + mul(slice_rot, float3(face_verts_2d[face_vert_idx] * 0.5, -0.5) * voxel_size);
        const float4 vs_pos = mul(frame_constants.view_constants.world_to_view, float4(ws_pos, 1.0));
        const float4 cs_pos = mul(frame_constants.view_constants.view_to_sample, vs_pos);

        float3 color = 0.1.xxx;

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