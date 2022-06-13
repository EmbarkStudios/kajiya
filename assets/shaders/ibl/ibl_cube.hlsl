#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/cube_map.hlsl"
#include "../inc/samplers.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] RWTexture2DArray<float4> output_tex;
[[vk::binding(2)]] cbuffer _ {
    uint cube_width;
};

// https://learnopengl.com/PBR/IBL/Diffuse-irradiance
float2 direction_to_spherical_map_uv(float3 v) {
    float2 uv = float2(atan2(v.z, v.x), asin(v.y));
    uv *= float2(0.1591, 0.3183);
    uv += 0.5;
    uv.y = 1 - uv.y;
    return uv;
}

[numthreads(8, 8, 1)]
void main(in uint3 px : SV_DispatchThreadID) {
    uint face = px.z;
    float2 uv = (px.xy + 0.5) / cube_width;
    float3 dir = normalize(mul(CUBE_MAP_FACE_ROTATIONS[face], float3(uv * 2 - 1, -1.0)));

    uv = direction_to_spherical_map_uv(dir);

    float3 output = input_tex.SampleLevel(sampler_llr, uv, 0).rgb;

    output_tex[px] = float4(frame_constants.pre_exposure * output, 1);
}
