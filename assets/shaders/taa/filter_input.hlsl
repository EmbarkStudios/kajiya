#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/math.hlsl"
#include "taa_common.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] RWTexture2D<float3> output_tex;
[[vk::binding(3)]] RWTexture2D<float3> dev_output_tex;

struct InputRemap {
    static InputRemap create() {
        InputRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return float4(rgb_to_ycbcr(decode_rgb(v.rgb)), 1);
    }
};

struct FilteredInput {
    float3 clamped_ex;
    float3 var;
};

FilteredInput filter_input_inner(uint2 px, float center_depth, float luma_cutoff, float depth_scale) {
    float3 iex = 0;
    float3 iex2 = 0;
    float iwsum = 0;

    float3 clamped_iex = 0;
    float clamped_iwsum = 0;

    InputRemap input_remap = InputRemap::create();

    const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const int2 spx_offset = int2(x, y);
            const float distance_w = exp(-(0.8 / (k * k)) * dot(spx_offset, spx_offset));

            const int2 spx = int2(px) + spx_offset;
            float3 s = input_remap.remap(input_tex[spx]).rgb;

            const float depth = depth_tex[spx];
            float w = 1;
            w *= exp2(-min(16, depth_scale * inverse_depth_relative_diff(center_depth, depth)));
            w *= distance_w;
            w *= pow(saturate(luma_cutoff / s.x), 8);

            clamped_iwsum += w;
            clamped_iex += s * w;

            iwsum += 1;
            iex += s;
            iex2 += s * s;
        }
    }

    clamped_iex /= clamped_iwsum;

    iex /= iwsum;
    iex2 /= iwsum;

    FilteredInput res;
    res.clamped_ex = clamped_iex;
    res.var = max(0, iex2 - iex * iex);
    
    return res;
}

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    const float center_depth = depth_tex[px];

    // Filter the input, with a cross-bilateral weight based on depth
    FilteredInput filtered_input = filter_input_inner(px, center_depth, 1e10, 200);

    // Filter the input again, but add another cross-bilateral weight, reducing the weight of
    // inputs brighter than the just-estimated luminance mean. This clamps bright outliers in the input.
    FilteredInput clamped_filtered_input = filter_input_inner(px, center_depth, filtered_input.clamped_ex.x * 1.001, 200);

    output_tex[px] = clamped_filtered_input.clamped_ex;
    dev_output_tex[px] = sqrt(filtered_input.var);
}
