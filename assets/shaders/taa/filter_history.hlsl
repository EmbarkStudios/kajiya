#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/math.hlsl"
#include "taa_common.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] RWTexture2D<float3> output_tex;
[[vk::binding(2)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

float3 filter_input(float2 uv, float luma_cutoff, int kernel_radius) {
    float3 iex = 0;
    float iwsum = 0;

    // Note: + epislon to counter precision loss, which manifests itself
    // as bad rounding in a 2x upscale, showing stair-stepping artifacts.
    int2 src_px = int2(floor(uv * input_tex_size.xy + 1e-3));

    const int k = kernel_radius;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const int2 spx_offset = int2(x, y);
            const int2 spx = int2(src_px) + spx_offset;

            // TODO: consider a weight based on uv diffs between the low-res
            // output `uv` and the low-res input `spx`.
            const float distance_w = exp(-(0.8 / (k * k)) * dot(spx_offset, spx_offset));

            float3 s = rgb_to_ycbcr(input_tex[spx].rgb);

            float w = 1;
            w *= distance_w;
            w *= pow(saturate(luma_cutoff / s.x), 8);

            iwsum += w;
            iex += s * w;
        }
    }

    return iex / iwsum;
}

void filter_history(uint2 px, int kernel_radius) {
    float2 uv = get_uv(px, output_tex_size);
    float filtered_luma = filter_input(uv, 1e10, kernel_radius).x;
    output_tex[px] = filter_input(uv, filtered_luma * 1.001, kernel_radius);
}

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    if (input_tex_size.x / output_tex_size.x > 1.75) {
        // If we're upscaling, history is at a higher resolution than
        // the new frame, so we need to filter history more.
        filter_history(px, 2);
    } else {
        filter_history(px, 1);
    }
}
