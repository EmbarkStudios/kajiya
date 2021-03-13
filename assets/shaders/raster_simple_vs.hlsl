#include "inc/frame_constants.hlsl"
#include "inc/mesh.hlsl"
#include "inc/bindless.hlsl"

[[vk::push_constant]]
struct {
    uint mesh_index;
    float instance_position[3];
} push_constants;

struct VsOut {
	float4 position: SV_Position;
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float2 uv: TEXCOORD1;
    [[vk::location(2)]] float3 normal: TEXCOORD2;
    [[vk::location(3)]] nointerpolation uint material_id: TEXCOORD3;
    [[vk::location(4)]] float3 tangent: TEXCOORD4;
    [[vk::location(5)]] float3 bitangent: TEXCOORD5;
    [[vk::location(6)]] float3 pos: TEXCOORD6;
};

VsOut main(uint vid: SV_VertexID, uint instance_index: SV_InstanceID) {
    VsOut vsout;

    const Mesh mesh = meshes[push_constants.mesh_index];

    // TODO: replace with Load<float4> once there's a fast path for NV
    // https://github.com/microsoft/DirectXShaderCompiler/issues/2193
    VertexPacked vp = VertexPacked(asfloat(vertices.Load4(vid * sizeof(float4) + mesh.vertex_core_offset)));
    Vertex v = unpack_vertex(vp);

    /*float4 v_color =
        mesh.vertex_aux_offset != 0
            ? asfloat(vertices.Load4(vid * sizeof(float4) + mesh.vertex_aux_offset))
            : 1.0.xxxx;*/

    float4 v_tangent_packed =
        mesh.vertex_tangent_offset != 0
            ? asfloat(vertices.Load4(vid * sizeof(float4) + mesh.vertex_tangent_offset))
            : float4(1, 0, 0, 1);            

    float2 uv = asfloat(vertices.Load2(vid * sizeof(float2) + mesh.vertex_uv_offset));
    uint material_id = vertices.Load(vid * sizeof(uint) + mesh.vertex_mat_offset);

    float3 ws_pos = v.position + float3(push_constants.instance_position);
    float4 vs_pos = mul(frame_constants.view_constants.world_to_view, float4(ws_pos, 1.0));
    float4 cs_pos = mul(frame_constants.view_constants.view_to_sample, vs_pos);

    vsout.position = cs_pos;
    vsout.color = 1.0;
    vsout.uv = uv;
    vsout.normal = v.normal;
    vsout.material_id = material_id;
    vsout.tangent = v_tangent_packed.xyz;
    vsout.bitangent = normalize(cross(v.normal, vsout.tangent) * v_tangent_packed.w);

    vsout.pos = vs_pos.xyz / vs_pos.w;

    return vsout;
}