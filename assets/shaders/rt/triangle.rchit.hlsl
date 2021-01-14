#include "../inc/mesh.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/bindless.hlsl"
#include "../inc/rt.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(1, 0)]] SamplerState sampler_llr;

struct RayHitAttrib {
    float2 bary;
};

[shader("closesthit")]
void main(inout GbufferRayPayload payload : SV_RayPayload, in RayHitAttrib attrib : SV_IntersectionAttributes) {
    float3 hit_point = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    float3 barycentrics = float3(1.0 - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);
    barycentrics.z += float(meshes[0].vertex_core_offset) * 1e-20;
    barycentrics.z += float(vertices.Load(0)) * 1e-20;
    barycentrics.z += material_textures[0][uint2(0, 0)].x * 1e-20;

    Mesh mesh = meshes[InstanceIndex()];

    // Indices of the triangle
    uint3 ind = uint3(
        vertices.Load((PrimitiveIndex() * 3 + 0) * sizeof(uint) + mesh.index_offset),
        vertices.Load((PrimitiveIndex() * 3 + 1) * sizeof(uint) + mesh.index_offset),
        vertices.Load((PrimitiveIndex() * 3 + 2) * sizeof(uint) + mesh.index_offset)
    );

    Vertex v0 = unpack_vertex(VertexPacked(asfloat(vertices.Load4(ind.x * sizeof(float4) + mesh.vertex_core_offset))));
    Vertex v1 = unpack_vertex(VertexPacked(asfloat(vertices.Load4(ind.y * sizeof(float4) + mesh.vertex_core_offset))));
    Vertex v2 = unpack_vertex(VertexPacked(asfloat(vertices.Load4(ind.z * sizeof(float4) + mesh.vertex_core_offset))));
    float3 normal = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;

    float2 uv0 = asfloat(vertices.Load2(ind.x * sizeof(float2) + mesh.vertex_uv_offset));
    float2 uv1 = asfloat(vertices.Load2(ind.y * sizeof(float2) + mesh.vertex_uv_offset));
    float2 uv2 = asfloat(vertices.Load2(ind.z * sizeof(float2) + mesh.vertex_uv_offset));
    float2 uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

    float3 color = 1;

    uint material_id = vertices.Load(ind.x * sizeof(uint) + mesh.vertex_mat_offset);
    MeshMaterial material = vertices.Load<MeshMaterial>(mesh.mat_data_offset + material_id * sizeof(MeshMaterial));

    Texture2D albedo_tex = material_textures[NonUniformResourceIndex(material.albedo_map)];
    float3 albedo = albedo_tex.SampleLevel(sampler_llr, uv, 0).xyz * float4(material.base_color_mult).xyz * color.xyz;

    Texture2D spec_tex = material_textures[NonUniformResourceIndex(material.spec_map)];
    float4 metalness_roughness = spec_tex.SampleLevel(sampler_llr, uv, 0);

    GbufferData gbuffer;
    gbuffer.albedo = albedo;
    gbuffer.normal = normal;
    gbuffer.roughness = clamp(material.roughness_mult * metalness_roughness.x, 0.01, 0.99);
    gbuffer.metalness = lerp(metalness_roughness.z, 1.0, material.metalness_factor);

    payload.gbuffer_packed = gbuffer.pack();
    payload.t = RayTCurrent();
}
