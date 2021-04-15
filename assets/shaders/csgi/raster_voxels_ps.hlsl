#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/gbuffer.hlsl"

struct PsIn {
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};

struct PsOut {
    float3 geometric_normal: SV_TARGET0;
    float4 gbuffer: SV_TARGET1;
    float4 velocity: SV_TARGET2;
};

PsOut main(PsIn ps) {
    GbufferData gbuffer = GbufferData::create_zero();
    gbuffer.albedo = ps.color.rgb;
    gbuffer.normal = ps.normal;
    gbuffer.roughness = 0.5;
    gbuffer.metalness = 0.0;

    float3 geometric_normal = mul(frame_constants.view_constants.world_to_view, float4(ps.normal, 0)).xyz;

    PsOut ps_out;
    ps_out.geometric_normal = geometric_normal * 0.5 + 0.5;
    ps_out.gbuffer = asfloat(gbuffer.pack().data0);
    ps_out.velocity = 0.0;
    return ps_out;
}
