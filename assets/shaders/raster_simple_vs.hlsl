#include "inc/frame_constants.hlsl"

struct VertexPacked {
	float4 data0;
};

struct Mesh {
    uint vertex_core_offset;
    uint vertex_uv_offset;
    uint vertex_aux_offset;
};

[[vk::binding(0, 1)]] StructuredBuffer<Mesh> meshes;
[[vk::binding(1, 1)]] ByteAddressBuffer vertices;

struct Vertex {
    float3 position;
    float3 normal;
};

float3 unpack_unit_direction_11_10_11(uint pck) {
    return float3(
        float(pck & ((1u << 11u)-1u)) * (2.0f / float((1u << 11u)-1u)) - 1.0f,
        float((pck >> 11u) & ((1u << 10u)-1u)) * (2.0f / float((1u << 10u)-1u)) - 1.0f,
        float((pck >> 21u)) * (2.0f / float((1u << 11u)-1u)) - 1.0f
    );
}

uint floatBitsToUint(float a) {
    return asuint(a);
}

Vertex unpack_vertex(VertexPacked p) {
    Vertex res;
    res.position = p.data0.xyz;
    res.normal = unpack_unit_direction_11_10_11(floatBitsToUint(p.data0.w));
    return res;
}

struct VsOut {
	float4 position: SV_Position;
    [[vk::location(0)]] float4 color: COLOR0;
    [[vk::location(1)]] float2 uv: COLOR1;
};

VsOut main(uint vid: SV_VertexID) {
    VsOut vsout;

    Mesh mesh = meshes[0];

    VertexPacked vp;
    // TODO: replace with Load<float4> once there's a fast path for NV
    // https://github.com/microsoft/DirectXShaderCompiler/issues/2193
    vp.data0 = asfloat(vertices.Load4(vid * 16 + mesh.vertex_core_offset));
    Vertex v = unpack_vertex(vp);

    float4 v_color = asfloat(vertices.Load4(vid * 16 + mesh.vertex_aux_offset));
    float2 uv = asfloat(vertices.Load2(vid * 8 + mesh.vertex_uv_offset));

    float4 vs_pos = mul(frame_constants.view_constants.world_to_view, float4(v.position, 1.0));
    float4 cs_pos = mul(frame_constants.view_constants.view_to_sample, vs_pos);

    vsout.position = cs_pos;
    //vsout.color = v_color;//float4(v.normal * 0.5 + 0.5, 1);
    vsout.color = float4(v.normal * 0.5 + 0.5, 1);
    vsout.uv = uv;

    return vsout;
}