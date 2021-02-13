#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/color.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture3D<float4> dir0_tex;
[[vk::binding(3)]] RWTexture2D<float4> out_tex;

[[vk::binding(4)]] cbuffer _ {
    float4 out_tex_size;
};

#include "common.hlsl"

[numthreads(8, 8, 1)]
void main(
    uint2 px: SV_DispatchThreadID
) {
    const float2 uv = get_uv(px, out_tex_size);
    out_tex[px] = float4(0, 0, 0, 1);

    const float4 gbuffer_packed = gbuffer_tex[px];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        return;
    }

    const float z_over_w = depth_tex[px];
    const float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    const float4 pt_vs = mul(frame_constants.view_constants.sample_to_view, pt_cs);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, pt_vs);
    pt_ws /= pt_ws.w;

    const float pt_depth = -pt_vs.z / pt_vs.w;

    // TODO: nuke
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();

    float3 irradiance = frac(pt_ws.xyz * GI_VOLUME_SCALE);

    //irradiance = dir0_tex.SampleLevel(sampler_nnc, pt_ws.xyz, 0).xyz;
    float3 noff = gbuffer.normal;
    //noff *= 1.0 / max(abs(noff.x), max(abs(noff.y), abs(noff.z)));
    //noff = trunc(noff);

    //float3 vol_pos = pt_ws.xyz + float3(0, -1, 0) + noff * 1.1 / 16.0;
    //irradiance = dir0_tex[(mul(slice_rot, vol_pos * 16) + float3(0, 0, 0)) + 16].rgb;

    float3 vol_pos = pt_ws.xyz - GI_VOLUME_CENTER + noff * 1.1 * GI_VOLUME_SCALE;
    irradiance = dir0_tex[int3(mul(slice_rot, vol_pos / GI_VOLUME_SCALE) + GI_VOLUME_DIMS / 2)].rgb;
    //irradiance = dir0_tex.SampleLevel(sampler_lnc, mul(slice_rot, vol_pos * 0.5 + 0.5), 0).rgb;
    //irradiance *= saturate(0.05 + dot(mul(slice_rot, gbuffer.normal), float3(0, 0, -1)));

    //if (uv.x > 0.5)
    {
        out_tex[px] = float4(irradiance, 1);
    }
}
