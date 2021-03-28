#include "../inc/samplers.hlsl"
#include "../inc/color.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> reprojection_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;
[[vk::binding(3)]] cbuffer _ {
    float4 output_tex_size;
};


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    
    float4 center = input_tex[px];
    float4 reproj = reprojection_tex[px];

    // TODO: validity-constrained interpolation
    float4 history = input_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);

    output_tex[px] = history;
}
