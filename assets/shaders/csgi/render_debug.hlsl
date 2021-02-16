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
[[vk::binding(2)]] Texture3D<float4> cascade0_tex;
[[vk::binding(3)]] Texture3D<float4> alt_cascade0_tex;
[[vk::binding(4)]] RWTexture2D<float4> out_tex;
[[vk::binding(5)]] cbuffer _ {
    float4 out_tex_size;
    float4 SLICE_DIRS[16];
};


#include "lookup.hlsl"

[numthreads(8, 8, 1)]
void main(
    uint2 px: SV_DispatchThreadID
) {
    const float2 uv = get_uv(px, out_tex_size);

    const float4 gbuffer_packed = gbuffer_tex[px];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        out_tex[px] = float4(0, 0, 0, 1);
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

    CsgiLookupParams gi_lookup_params;
    gi_lookup_params.use_grid_linear_fetch = true;
    gi_lookup_params.use_pretrace = false;
    gi_lookup_params.debug_slice_idx = -1;
    //gi_lookup_params.slice_dirs = SLICE_DIRS;
    //gi_lookup_params.cascade0_tex = cascade0_tex;
    //gi_lookup_params.alt_cascade0_tex = alt_cascade0_tex;

    irradiance = lookup_csgi(pt_ws.xyz, gbuffer.normal, gi_lookup_params);

    //irradiance = cascade0_tex.SampleLevel(sampler_lnc, mul(slice_rot, vol_pos * 0.5 + 0.5), 0).rgb;
    //irradiance *= saturate(0.05 + dot(mul(slice_rot, gbuffer.normal), float3(0, 0, -1)));

    //if (uv.x > 0.5)
    {
        out_tex[px] += float4(irradiance * lerp(gbuffer.albedo, 0.0, gbuffer.metalness), 0);
    }

    //out_tex[px] = float4(irradiance, 1);
}
