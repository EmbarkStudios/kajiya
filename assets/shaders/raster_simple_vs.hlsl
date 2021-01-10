#include "inc/frame_constants.hlsl"
#include "inc/mesh.hlsl"

[[vk::binding(0, 1)]] StructuredBuffer<Mesh> meshes;
[[vk::binding(1, 1)]] ByteAddressBuffer vertices;

struct VsOut {
	float4 position: SV_Position;
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float2 uv: TEXCOORD1;
    [[vk::location(2)]] float3 normal: TEXCOORD2;
    [[vk::location(3)]] nointerpolation uint material_id: TEXCOORD3;
};

VsOut main(uint vid: SV_VertexID) {
    VsOut vsout;

    Mesh mesh = meshes[0];

    VertexPacked vp;
    // TODO: replace with Load<float4> once there's a fast path for NV
    // https://github.com/microsoft/DirectXShaderCompiler/issues/2193
    vp.data0 = asfloat(vertices.Load4(vid * sizeof(float4) + mesh.vertex_core_offset));
    Vertex v = unpack_vertex(vp);

    /*float4 v_color =
        mesh.vertex_aux_offset != 0
            ? asfloat(vertices.Load4(vid * sizeof(float4) + mesh.vertex_aux_offset))
            : 1.0.xxxx;*/

    float2 uv = asfloat(vertices.Load2(vid * sizeof(float2) + mesh.vertex_uv_offset));
    uint material_id = vertices.Load(vid * sizeof(uint) + mesh.vertex_mat_offset);

    float4 vs_pos = mul(frame_constants.view_constants.world_to_view, float4(v.position, 1.0));
    float4 cs_pos = mul(frame_constants.view_constants.view_to_sample, vs_pos);

    vsout.position = cs_pos;
    vsout.color = 1.0;
    vsout.uv = uv;
    vsout.normal = v.normal;
    vsout.material_id = material_id;

    return vsout;
}