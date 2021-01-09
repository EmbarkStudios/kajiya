#include "inc/mesh.hlsl"

[[vk::binding(0, 1)]] StructuredBuffer<Mesh> meshes;
[[vk::binding(1, 1)]] ByteAddressBuffer vertices;

[[vk::binding(0, 3)]] Texture2D material_textures[];
SamplerState sampler_llr;

struct PsIn {
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float2 uv: TEXCOORD1;
    [[vk::location(2)]] nointerpolation uint material_id: TEXCOORD2;
};

float4 main(PsIn ps/*, float4 cs_pos: SV_Position*/): SV_TARGET {
    Mesh mesh = meshes[0];
    MeshMaterial material = vertices.Load<MeshMaterial>(mesh.mat_data_offset + ps.material_id * sizeof(MeshMaterial));

    Texture2D tex = material_textures[NonUniformResourceIndex(material.albedo_map)];
    float4 col = tex.Sample(sampler_llr, ps.uv) * float4(material.base_color_mult);
    return col;

    return ps.color;
}
