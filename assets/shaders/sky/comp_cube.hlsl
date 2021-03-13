#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"

[[vk::binding(0)]] RWTexture2DArray<float4> output_tex;

static const float3x3 FACE_ROTATIONS[6] = {
    float3x3(0,0,-1, 0,-1,0, -1,0,0),   // right
    float3x3(0,0,1, 0,-1,0, 1,0,0),     // left

    float3x3(1,0,0, 0,0,-1, 0,1,0),     // top
    float3x3(1,0,0, 0,0,1, 0,-1,0),     // bottom

    float3x3(1,0,0, 0,-1,0, 0,0,-1),    // back
    float3x3(-1,0,0, 0,-1,0, 0,0,1),    // front
};

[numthreads(8, 8, 1)]
void main(in uint3 px : SV_DispatchThreadID) {
    uint face = px.z;
    float2 uv = (px.xy + 0.5) / 32;
    float3 dir = normalize(mul(FACE_ROTATIONS[face], float3(uv * 2 - 1, -1.0)));

    //float3 output = dir * 0.5 + 0.5;
    float3 output = atmosphere_default(dir, SUN_DIRECTION);

    output_tex[px] = float4(output, 1);
}
