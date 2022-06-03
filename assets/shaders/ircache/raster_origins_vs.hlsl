#include "../inc/frame_constants.hlsl"
#include "../inc/math.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked
#include "ircache_constants.hlsl"

[[vk::binding(0)]] ByteAddressBuffer ircache_meta_buf;
[[vk::binding(1)]] ByteAddressBuffer ircache_grid_meta_buf;
[[vk::binding(2)]] StructuredBuffer<uint> ircache_life_buf;
[[vk::binding(3)]] StructuredBuffer<VertexPacked> ircache_spatial_buf;

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

    const float3 cube_center = unpack_vertex(ircache_spatial_buf[entry_idx]).position;

    if (is_ircache_entry_life_valid(ircache_life_buf[entry_idx])) {
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

        const float cube_size = 0.02;

        const float3 ws_pos = cube_center + mul(slice_rot, float3(face_verts_2d[face_vert_idx] * 0.5, -0.5) * cube_size);
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