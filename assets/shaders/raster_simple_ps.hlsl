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
    [[vk::location(6)]] float3 vs_pos: TEXCOORD6;
    [[vk::location(7)]] float3 prev_vs_pos: TEXCOORD7;
};

[[vk::push_constant]]
struct {
    uint mesh_index;
    float instance_position[3];
    float prev_instance_position[3];
} push_constants;

struct PsOut {
    float4 gbuffer: SV_TARGET0;
    float2 velocity: SV_TARGET1;
};

PsOut main(PsIn ps) {
    Mesh mesh = meshes[push_constants.mesh_index];
    MeshMaterial material = vertices.Load<MeshMaterial>(mesh.mat_data_offset + ps.material_id * sizeof(MeshMaterial));

    float2 albedo_uv = transform_material_uv(material, ps.uv, 0);
    Texture2D albedo_tex = bindless_textures[NonUniformResourceIndex(material.albedo_map)];
    //float3 albedo = albedo_tex.SampleLevel(sampler_llr, ps.uv, 0).xyz * float4(material.base_color_mult).xyz * ps.color.xyz;
    float3 albedo = albedo_tex.SampleBias(sampler_llr, albedo_uv, -0.5).xyz * float4(material.base_color_mult).xyz * ps.color.xyz;

    float2 spec_uv = transform_material_uv(material, ps.uv, 2);
    Texture2D spec_tex = bindless_textures[NonUniformResourceIndex(material.spec_map)];
    const float4 metalness_roughness = spec_tex.SampleBias(sampler_llr, spec_uv, -0.5);

    float roughness = clamp(material.roughness_mult * metalness_roughness.y, 1e-3, 1.0);
    //roughness = 0.01;
    float metalness = metalness_roughness.z * material.metalness_factor;//lerp(metalness_roughness.z, 1.0, material.metalness_factor);

    //albedo *= lerp(0.75, 1.0, metalness);

    Texture2D normal_tex = bindless_textures[NonUniformResourceIndex(material.normal_map)];
    const float3 ts_normal = normal_tex.SampleBias(sampler_llr, ps.uv, -0.5).xyz * 2.0 - 1.0;

    float3 normal = ps.normal;
    if (dot(ps.bitangent, ps.bitangent) > 0.0) {
        float3x3 tbn = float3x3(ps.tangent, ps.bitangent, ps.normal);
        normal = mul(ts_normal, tbn);
    }
    normal = normalize(normal);

    if (!true) {
        // Derive normal from depth
        float3 d1 = ddx(ps.vs_pos);
        float3 d2 = ddy(ps.vs_pos);
        normal = normalize(mul(frame_constants.view_constants.view_to_world, float4(cross(d2,d1), 0)).xyz); // this normal is dp/du X dp/dv
    }

    //albedo = float3(0.966653, 0.802156, 0.323968); // Au from Mitsuba
    //metalness = 1;
    //roughness = 0.2;

    GbufferData gbuffer;
    gbuffer.albedo = albedo;
    gbuffer.normal = normal;
    gbuffer.roughness = roughness;
    gbuffer.metalness = metalness;

    PsOut ps_out;
    ps_out.gbuffer = asfloat(gbuffer.pack().data0);

    float4 cs_pos = mul(frame_constants.view_constants.view_to_sample, float4(ps.vs_pos, 1));
    float4 prev_cs_pos = mul(frame_constants.view_constants.view_to_sample, float4(ps.prev_vs_pos, 1));
    float2 uv_pos = cs_to_uv(cs_pos.xy / cs_pos.w);
    float2 prev_uv_pos = cs_to_uv(prev_cs_pos.xy / prev_cs_pos.w);

    ps_out.velocity = prev_uv_pos - uv_pos;

    return ps_out;
}
