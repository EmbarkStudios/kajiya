#include "../inc/mesh.hlsl"
#include "../inc/bindless.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

struct Payload {
    float3 hitValue;
};

struct Attribute {
    float2 bary;
};

struct ShadowPayload {
    bool is_shadowed;
};

[shader("closesthit")]
void main(inout Payload payload : SV_RayPayload, in Attribute attribs : SV_IntersectionAttributes) {
    float3 hit_point = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    RayDesc shadow_ray;
    shadow_ray.Origin = hit_point;
    shadow_ray.Direction = normalize(float3(1, 1, 1));
    shadow_ray.TMin = 0.001;
    shadow_ray.TMax = 100000.0;

    ShadowPayload shadow_payload;
    shadow_payload.is_shadowed = true;
    TraceRay(
        acceleration_structure,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xff, 0, 0, 0, shadow_ray, shadow_payload
    );

    if (!shadow_payload.is_shadowed) {
        float3 barycentrics = float3(1.0 - attribs.bary.x - attribs.bary.y, attribs.bary.x, attribs.bary.y);
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
        Vertex v1 = unpack_vertex(VertexPacked(asfloat(vertices.Load4(ind.x * sizeof(float4) + mesh.vertex_core_offset))));
        Vertex v2 = unpack_vertex(VertexPacked(asfloat(vertices.Load4(ind.x * sizeof(float4) + mesh.vertex_core_offset))));

        float3 normal = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;

        payload.hitValue = normal * 0.5 + 0.5;
    } else {
        payload.hitValue = 0.01.xxx;
    }
}
