#include "inc/frame_constants.hlsl"
#include "inc/mesh.hlsl"
#include "inc/pack_unpack.hlsl"
#include "inc/bindless.hlsl"
#include "inc/gbuffer.hlsl"

SamplerState sampler_llr;

struct PsIn {
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float2 uv: TEXCOORD1;
    [[vk::location(2)]] float3 normal: TEXCOORD2;
    [[vk::location(3)]] nointerpolation uint material_id: TEXCOORD3;
    [[vk::location(4)]] float3 tangent: TEXCOORD4;
    [[vk::location(5)]] float3 bitangent: TEXCOORD5;
    //[[vk::location(4)]] float3 pos: TEXCOORD4;
};

[[vk::push_constant]]
struct {
    uint mesh_index;
    float instance_position[3];
} push_constants;


float4 main(PsIn ps/*, float4 cs_pos: SV_Position*/): SV_TARGET {
    Mesh mesh = meshes[push_constants.mesh_index];
    MeshMaterial material = vertices.Load<MeshMaterial>(mesh.mat_data_offset + ps.material_id * sizeof(MeshMaterial));

    //float3 d1 = ddx(ps.pos);
    //float3 d2 = ddy(ps.pos);
    //normal = normalize(mul(frame_constants.view_constants.view_to_world, float4(cross(d2,d1), 0)).xyz); // this normal is dp/du X dp/dv

    float2 albedo_uv = transform_material_uv(material, ps.uv, 0);
    Texture2D albedo_tex = bindless_textures[NonUniformResourceIndex(material.albedo_map)];
    //float3 albedo = albedo_tex.SampleLevel(sampler_llr, ps.uv, 0).xyz * float4(material.base_color_mult).xyz * ps.color.xyz;
    float3 albedo = albedo_tex.Sample(sampler_llr, albedo_uv).xyz * float4(material.base_color_mult).xyz * ps.color.xyz;

    float2 spec_uv = transform_material_uv(material, ps.uv, 2);
    Texture2D spec_tex = bindless_textures[NonUniformResourceIndex(material.spec_map)];
    const float4 metalness_roughness = spec_tex.Sample(sampler_llr, spec_uv);

    float roughness = clamp(material.roughness_mult * metalness_roughness.y, 1e-3, 1.0);
    //roughness = 0.01;
    float metalness = metalness_roughness.z * material.metalness_factor;//lerp(metalness_roughness.z, 1.0, material.metalness_factor);

    Texture2D normal_tex = bindless_textures[NonUniformResourceIndex(material.normal_map)];
    const float3 ts_normal = normal_tex.Sample(sampler_llr, ps.uv).xyz * 2.0 - 1.0;

    float3 normal = ps.normal;
    if (dot(ps.bitangent, ps.bitangent) > 0.0) {
        float3x3 tbn = float3x3(ps.tangent, ps.bitangent, ps.normal);
        normal = mul(ts_normal, tbn);
    }
    normal = normalize(normal);

    //albedo = float3(0.966653, 0.802156, 0.323968); // Au from Mitsuba
    //metalness = 1;
    //roughness = 0.6;

    GbufferData gbuffer;
    gbuffer.albedo = albedo;
    gbuffer.normal = normal;
    gbuffer.roughness = roughness;
    gbuffer.metalness = metalness;
    return asfloat(gbuffer.pack().data0);
}
