#include "inc/samplers.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tail_tex;
[[vk::binding(1)]] Texture2D<float4> input_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;
[[vk::binding(3)]] cbuffer _ {
    uint output_extent_x;
    uint output_extent_y;
    float self_weight;
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID, uint2 px_within_group: SV_GroupThreadID, uint2 group_id: SV_GroupID) {
    float4 pyramid_col = input_tail_tex[px];

#if 1
    // TODO: do a small Gaussian blur instead of this nonsense

    static const int k = 1;
    float4 self_col = 0;
    float wt_sum = 0;

    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float2 uv = (float2(px) + 0.5 + float2(x, y)) / float2(output_extent_x, output_extent_y);
            self_col += input_tex.SampleLevel(sampler_lnc, uv, 0);
        }
    }

    self_col /= (2 * k + 1) * (2 * k + 1);
#else
    float2 uv = (px + 0.5) / float2(output_extent_x, output_extent_y);
    //float4 self_col = input_tex[px / 2];
    float4 self_col = input_tex.SampleLevel(sampler_lnc, uv, 0);
#endif

    const float exponential_falloff = 0.5;

    // BUG: when `self_weight` is 1.0, the `w` here should be 1.0, not `exponential_falloff`
    output_tex[px] = lerp(self_col, pyramid_col, self_weight * exponential_falloff);
}
