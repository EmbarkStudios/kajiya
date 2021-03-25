#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/gbuffer.hlsl"

struct PsIn {
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD2;
};

struct PsOut {
    float4 gbuffer: SV_TARGET0;
    float2 velocity: SV_TARGET1;
};

PsOut main(PsIn ps) {
    GbufferData gbuffer;
    gbuffer.albedo = ps.color.rgb;
    gbuffer.normal = ps.normal;
    gbuffer.roughness = 0.5;
    gbuffer.metalness = 0.0;

    PsOut ps_out;
    ps_out.gbuffer = asfloat(gbuffer.pack().data0);
    ps_out.velocity = 0.0.xx;
    return ps_out;
}
