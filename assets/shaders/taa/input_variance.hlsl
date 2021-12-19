#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"
#include "../inc/image.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/unjitter_taa.hlsl"
#include "../inc/soft_color_clamp.hlsl"
#include "taa_common.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] RWTexture2D<float> output_tex;
[[vk::binding(3)]] cbuffer _ {
    float4 input_tex_size;
};

struct InputRemap {
    static InputRemap create() {
        InputRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return float4(rgb_to_ycbcr(decode_rgb(v.rgb)), 1);
    }
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    const float3 input = rgb_to_ycbcr(decode_rgb(input_tex[px].rgb));
    
    float3 iex = 0;
    float3 iex2 = 0;
    float iwsum = 0;
    {
        int k = 1;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float3 s = rgb_to_ycbcr(decode_rgb(input_tex[px + int2(x, y)].rgb));
                float w = 1;
                iwsum += w;
                iex += s * w;
                iex2 += s * s * w;
            }
        }
    }

    iex /= iwsum;
    iex2 /= iwsum;

    float3 ivar = max(0, iex2 - iex * iex);
    //output_tex[px] = sqrt(ivar.x) / max(1e-5, iex.x);
    output_tex[px] = ivar.x;
}
