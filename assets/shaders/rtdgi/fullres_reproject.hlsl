#include "../inc/samplers.hlsl"
#include "../inc/color.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/bilinear.hlsl"
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
    float2 prev_uv = uv + reproj.xy;

    uint quad_reproj_valid_packed = uint(reproj.z * 15.0 + 0.5);
    float4 history = 0.0.xxxx;

    if (0 == quad_reproj_valid_packed) {
        // Everything invalid
    } else if (15 == quad_reproj_valid_packed) {
        history = input_tex.SampleLevel(sampler_lnc, prev_uv, 0);
    } else {
        float4 quad_reproj_valid = (quad_reproj_valid_packed & uint4(1, 2, 4, 8)) != 0;

        const Bilinear bilinear = get_bilinear_filter(prev_uv, output_tex_size.xy);
        float4 s00 = input_tex[int2(bilinear.origin) + int2(0, 0)];
        float4 s10 = input_tex[int2(bilinear.origin) + int2(1, 0)];
        float4 s01 = input_tex[int2(bilinear.origin) + int2(0, 1)];
        float4 s11 = input_tex[int2(bilinear.origin) + int2(1, 1)];

        float4 weights = get_bilinear_custom_weights(bilinear, quad_reproj_valid);

        if (dot(weights, 1.0) > 1e-5) {
            history = apply_bilinear_custom_weights(s00, s10, s01, s11, weights);
        }
    }

    output_tex[px] = history;
}
