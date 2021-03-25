#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/gbuffer.hlsl"

struct PsIn {
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD2;
};

float4 main(PsIn ps): SV_TARGET {
    GbufferData gbuffer;
    gbuffer.albedo = ps.color.rgb;
    gbuffer.normal = ps.normal;
    gbuffer.roughness = 0.5;
    gbuffer.metalness = 0.0;
    return asfloat(gbuffer.pack().data0);
}
