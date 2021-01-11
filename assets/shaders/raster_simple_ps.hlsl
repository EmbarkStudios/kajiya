#include "inc/mesh.hlsl"
#include "inc/pack_unpack.hlsl"
#include "inc/bindless.hlsl"

SamplerState sampler_llr;

struct PsIn {
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float2 uv: TEXCOORD1;
    [[vk::location(2)]] float3 normal: TEXCOORD2;
    [[vk::location(3)]] nointerpolation uint material_id: TEXCOORD3;
};

float4 main(PsIn ps/*, float4 cs_pos: SV_Position*/): SV_TARGET {
    Mesh mesh = meshes[0];
    MeshMaterial material = vertices.Load<MeshMaterial>(mesh.mat_data_offset + ps.material_id * sizeof(MeshMaterial));

    float3 normal = ps.normal;
    /*if (dot(v_bitangent, v_bitangent) > 0.0) {
        mat3 tbn = mat3(v_tangent, v_bitangent, v_normal);
        normal = tbn * ts_normal;
    }*/
    normal = normalize(normal);

    Texture2D albedo_tex = material_textures[NonUniformResourceIndex(material.albedo_map)];
    float3 albedo = albedo_tex.Sample(sampler_llr, ps.uv).xyz * float4(material.base_color_mult).xyz * ps.color.xyz;

    Texture2D spec_tex = material_textures[NonUniformResourceIndex(material.spec_map)];
    float4 metalness_roughness = spec_tex.Sample(sampler_llr, ps.uv);

    float roughness = clamp(material.roughness_mult * metalness_roughness.x, 0.01, 0.99);
    float metalness = lerp(metalness_roughness.z, 1.0, material.metalness_factor);
    float z_over_w = 1.0;

    float4 res = 0.0.xxxx;
    res.x = asfloat(pack_color_888(albedo));
    res.y = pack_normal_11_10_11(normal);
    res.z = roughness * roughness;      // UE4 remap
    res.w = metalness;

    return res;
}
