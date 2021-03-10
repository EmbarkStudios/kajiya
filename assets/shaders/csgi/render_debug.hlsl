#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/color.hlsl"
#include "../inc/math.hlsl"

#include "common.hlsl"


[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture3D<float4> csgi_cascade0_tex;
[[vk::binding(3)]] RWTexture2D<float4> out_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 out_tex_size;
    float4 CSGI_SLICE_DIRS[16];
    float4 CSGI_SLICE_CENTERS[16];
};


#include "lookup.hlsl"

[numthreads(8, 8, 1)]
void main(
    uint2 px: SV_DispatchThreadID
) {
#if 0
    out_tex[px] = 0;

    int frame_cycle = 100;
    int xoffset = sin(frame_constants.frame_index % frame_cycle / float(frame_cycle) * M_PI * 2) * 50;

    if (px.y == 500) {
        for (int i = 1; i <= 8; ++i) {
            if (px.x + xoffset == int(exp(i * 0.2) * 430) - 410) {
                float t =  pow(i * 0.5, 2.0) * 500 + 2200;
                float3 tint = blackbody_radiation(t);
                tint /= max(max(1e-5, tint.r), max(tint.g, tint.b));
                out_tex[px] = float4(pow(3.5, i-1) * 2.0 * tint, 1);
            }
        }
    }
#endif

#if 0

    const float2 uv = get_uv(px, out_tex_size);

    const float4 gbuffer_packed = gbuffer_tex[px];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        //out_tex[px] = float4(0, 0, 0, 1);
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

    float3 irradiance = 0.0.xxx;

    CsgiLookupParams params = CsgiLookupParams::make_default();
    //params.use_grid_linear_fetch = false;
    irradiance = lookup_csgi(pt_ws.xyz, gbuffer.normal, params);

    //irradiance = csgi_cascade0_tex.SampleLevel(sampler_lnc, mul(slice_rot, vol_pos * 0.5 + 0.5), 0).rgb;
    //irradiance *= saturate(0.05 + dot(mul(slice_rot, gbuffer.normal), float3(0, 0, -1)));

    //if (uv.x > 0.5)
    {
        //out_tex[px] += float4(irradiance * lerp(gbuffer.albedo, 0.0, gbuffer.metalness), 0);
    }

    out_tex[px] = float4(irradiance, 1);

#endif
}
