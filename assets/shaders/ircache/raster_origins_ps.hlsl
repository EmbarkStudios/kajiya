#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/gbuffer.hlsl"

struct PsIn {
    [[vk::location(0)]] float4 color: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};

struct PsOut {
    float3 color: SV_TARGET0;
};

PsOut main(PsIn ps) {
    float3 geometric_normal = mul(frame_constants.view_constants.world_to_view, float4(ps.normal, 0)).xyz;

    PsOut ps_out;
    ps_out.color = geometric_normal * 0.5 + 0.5;
    return ps_out;
}
