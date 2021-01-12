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

    payload.hitValue = barycentrics;
    //payload.hitValue.x = float(PrimitiveIndex() % 32) / 32.0;
}
