#include "../inc/mesh.hlsl"
#include "../inc/bindless.hlsl"

struct Payload {
    float3 hitValue;
};

struct Attribute {
    float2 bary;
};

[shader("closesthit")]
void main(inout Payload payload : SV_RayPayload, in Attribute attribs : SV_IntersectionAttributes) {
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
    //payload.hitValue.x = float(PrimitiveIndex() % 32) / 32.0;
}
