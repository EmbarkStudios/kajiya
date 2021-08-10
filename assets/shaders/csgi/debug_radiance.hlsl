#include "../inc/frame_constants.hlsl"
#include "../inc/samplers.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float3> csgi_direct_tex[CSGI_CASCADE_COUNT];
[[vk::binding(1)]] Texture3D<float3> csgi_indirect_tex[CSGI_CASCADE_COUNT];
[[vk::binding(2)]] Texture3D<float3> csgi_subray_indirect_tex[CSGI_CASCADE_COUNT];
[[vk::binding(3)]] RWTexture2D<float4> output_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 output_tex_size;
};

#define CSGI_LOOKUP_NO_DIRECT
#include "lookup.hlsl"
#include "subray_lookup.hlsl"

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv(uv);
    const float3 v = view_ray_context.ray_dir_ws();

#if 1
    float3 output = lookup_csgi(
        get_eye_position(),
        0.0.xxx,    // don't offset by any normal
        CsgiLookupParams::make_default()
            .with_sample_directional_radiance(v)
            //.with_directional_radiance_phong_exponent(8)
    );
#else
    float3 output = point_sample_csgi_subray_indirect(get_eye_position(), v);
#endif

    uint2 grid = px / 32;
    if ((grid.x | grid.y) & 1)
    {
        output_tex[px] = float4(output, 1.0);
    }
}
