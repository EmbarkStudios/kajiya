#include "inc/samplers.hlsl"
#include "inc/frame_constants.hlsl"
#include "inc/mesh.hlsl"
#include "inc/pack_unpack.hlsl"
#include "inc/bindless.hlsl"
#include "inc/gbuffer.hlsl"

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
    uint draw_index;
    uint mesh_index;
} push_constants;

struct InstanceTransform {
    row_major float3x4 current;
    row_major float3x4 previous;
};

[[vk::binding(0)]] StructuredBuffer<InstanceTransform> instance_transforms_dyn;

struct PsOut {
    float3 geometric_normal: SV_TARGET0;
    float4 gbuffer: SV_TARGET1;
    float4 velocity: SV_TARGET2;
};

PsOut main(PsIn ps) {
    Mesh mesh = meshes[push_constants.mesh_index];
    MeshMaterial material = vertices.Load<MeshMaterial>(mesh.mat_data_offset + ps.material_id * sizeof(MeshMaterial));

    float2 albedo_uv = transform_material_uv(material, ps.uv, 0);
    Texture2D albedo_tex = bindless_textures[NonUniformResourceIndex(material.albedo_map)];
    float4 albedo_texel = albedo_tex.SampleBias(sampler_llr, albedo_uv, -0.5);
    if (albedo_texel.a < 0.5) {
        discard;
    }

    float3 albedo = albedo_texel.xyz * float4(material.base_color_mult).xyz * ps.color.xyz;

    float2 spec_uv = transform_material_uv(material, ps.uv, 2);
    Texture2D spec_tex = bindless_textures[NonUniformResourceIndex(material.spec_map)];
    const float4 metalness_roughness = spec_tex.SampleBias(sampler_llr, spec_uv, -0.5);
    float perceptual_roughness = material.roughness_mult * metalness_roughness.y;
    float roughness = clamp(perceptual_roughness_to_roughness(perceptual_roughness), 1e-4, 1.0);
    float metalness = metalness_roughness.z * material.metalness_factor;

    Texture2D normal_tex = bindless_textures[NonUniformResourceIndex(material.normal_map)];
    const float3 ts_normal = normal_tex.SampleBias(sampler_llr, ps.uv, -0.5).xyz * 2.0 - 1.0;

    float3 normal_ws; {
        float3 normal_os = ps.normal;

        if (true) {
            if (dot(ps.bitangent, ps.bitangent) > 0.0) {
                float3x3 tbn = float3x3(ps.tangent, ps.bitangent, ps.normal);
                normal_os = mul(ts_normal, tbn);
            }
        }

        // Transform to world space
        normal_ws = normalize(mul(instance_transforms_dyn[push_constants.draw_index].current, float4(normal_os, 0.0)));
    }

    // Derive normal from depth
    float3 geometric_normal_vs; {
        float3 d1 = ddx(ps.vs_pos);
        float3 d2 = ddy(ps.vs_pos);
        geometric_normal_vs = normalize(cross(d2, d1));
    }
    float3 geometric_normal_ws = direction_view_to_world(geometric_normal_vs);

    // Fix invalid normals
    if (dot(normal_ws, geometric_normal_ws) < 0) {
        normal_ws *= -1;
        //normal_ws = geometric_normal_ws;
    }
    //normal_ws = geometric_normal_ws;

    float2 emissive_uv = transform_material_uv(material, ps.uv, 3);
    Texture2D emissive_tex = bindless_textures[NonUniformResourceIndex(material.emissive_map)];
    float3 emissive = 1.0.xxx
        * emissive_tex.SampleBias(sampler_llr, emissive_uv, -0.5).rgb
        * float3(material.emissive)
        * instance_dynamic_parameters_dyn[push_constants.draw_index].emissive_multiplier;

    //albedo = float3(0.966653, 0.802156, 0.323968); // Au from Mitsuba

    GbufferData gbuffer = GbufferData::create_zero();
    gbuffer.albedo = albedo;
    gbuffer.normal = normal_ws;
    gbuffer.roughness = roughness;
    gbuffer.metalness = metalness;
    gbuffer.emissive = emissive;

    PsOut ps_out;
    ps_out.geometric_normal = geometric_normal_vs * 0.5 + 0.5;
    ps_out.gbuffer = asfloat(gbuffer.pack().data0);
    ps_out.velocity = float4(ps.prev_vs_pos - ps.vs_pos, 0);

    return ps_out;
}
