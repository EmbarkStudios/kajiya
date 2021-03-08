#include "../inc/samplers.hlsl"
#include "../inc/mesh.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/bindless.hlsl"
#include "../inc/rt.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

struct RayHitAttrib {
    float2 bary;
};

float twice_triangle_area(float3 p0, float3 p1, float3 p2) {
    return length(cross(p1 - p0, p2 - p0));
}

float twice_uv_area(float2 t0, float2 t1, float2 t2) {
    return abs((t1.x - t0.x) * (t2.y - t0.y) - (t2.x - t0.x) * (t1.y - t0.y));
}

float compute_texture_lod(Texture2D tex, float triangle_constant, float3 ray_direction, float3 surf_normal, float cone_width) {
    uint w, h;
    tex.GetDimensions(w, h);

    float lambda = triangle_constant;
    lambda += log2(abs(cone_width));
    lambda += 0.5 * log2(float(w) * float(h));

    // TODO: This blurs a lot at grazing angles; do aniso.
    lambda -= log2(abs(dot(normalize(ray_direction), surf_normal)));

    return lambda;
}


[shader("closesthit")]
void main(inout GbufferRayPayload payload: SV_RayPayload, in RayHitAttrib attrib: SV_IntersectionAttributes) {
    float3 hit_point = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const float hit_dist = length(hit_point - WorldRayOrigin());

    float3 barycentrics = float3(1.0 - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);

    //Mesh mesh = meshes[InstanceIndex() / 2];
    Mesh mesh = meshes[InstanceID()];

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

    const float3 surf_normal = normalize(cross(v1.position - v0.position, v2.position - v0.position));

    float2 uv0 = asfloat(vertices.Load2(ind.x * sizeof(float2) + mesh.vertex_uv_offset));
    float2 uv1 = asfloat(vertices.Load2(ind.y * sizeof(float2) + mesh.vertex_uv_offset));
    float2 uv2 = asfloat(vertices.Load2(ind.z * sizeof(float2) + mesh.vertex_uv_offset));
    float2 uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

    const float cone_width = payload.cone_width * hit_dist;
    const float lod_triangle_constant = 0.5 * log2(twice_triangle_area(v0.position, v1.position, v2.position) / twice_uv_area(uv0, uv1, uv2));

    uint material_id = vertices.Load(ind.x * sizeof(uint) + mesh.vertex_mat_offset);
    MeshMaterial material = vertices.Load<MeshMaterial>(mesh.mat_data_offset + material_id * sizeof(MeshMaterial));

    float2 albedo_uv = transform_material_uv(material, uv, 0);
    Texture2D albedo_tex = bindless_textures[NonUniformResourceIndex(material.albedo_map)];
    float albedo_lod = 0;//compute_texture_lod(albedo_tex, lod_triangle_constant, WorldRayDirection(), surf_normal, cone_width);
    float3 albedo = albedo_tex.SampleLevel(sampler_llr, albedo_uv, albedo_lod).xyz * float4(material.base_color_mult).xyz;

    float2 spec_uv = transform_material_uv(material, uv, 2);
    Texture2D spec_tex = bindless_textures[NonUniformResourceIndex(material.spec_map)];
    float spec_lod = 0;//compute_texture_lod(spec_tex, lod_triangle_constant, WorldRayDirection(), surf_normal, cone_width);
    float4 metalness_roughness = spec_tex.SampleLevel(sampler_llr, spec_uv, spec_lod);

    GbufferData gbuffer;
    gbuffer.albedo = albedo;
    gbuffer.normal = normal;
    gbuffer.roughness = clamp(material.roughness_mult * metalness_roughness.y, 1e-3, 1.0);
    //gbuffer.metalness = lerp(metalness_roughness.z, 1.0, material.metalness_factor);
    gbuffer.metalness = metalness_roughness.z * material.metalness_factor;

    //gbuffer.albedo = float3(0.966653, 0.802156, 0.323968); // Au from Mitsuba
    //gbuffer.metalness = 1;
    //gbuffer.roughness = 1;

    payload.gbuffer_packed = gbuffer.pack();
    payload.t = RayTCurrent();
}
