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
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> reprojection_tex;
[[vk::binding(3)]] Texture2D<float> depth_tex;
[[vk::binding(4)]] Texture2D<float4> meta_history_tex;
[[vk::binding(5)]] Texture2D<float2> velocity_history_tex;
[[vk::binding(6)]] RWTexture2D<float> output_tex;
[[vk::binding(7)]] cbuffer _ {
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

float4 fetch_blurred_history(int2 px, int k, float sigma) {
    const float3 center = history_tex[px].rgb;

    float4 csum = 0;
    float wsum = 0;

    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 c = history_tex[px + int2(x, y)];
            float2 offset = float2(x, y) * sigma;
            float w = exp(-dot(offset, offset));
            float color_diff =
                linear_to_perceptual(calculate_luma(c.rgb))
                - linear_to_perceptual(calculate_luma(center));
            //w *= exp(-color_diff * color_diff * 100);
            csum += c * w;
            wsum += w;
        }
    }

    return csum / wsum;
}

struct HistoryRemap {
    static HistoryRemap create() {
        HistoryRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return float4(rgb_to_ycbcr(v.rgb), 1);
    }
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float input_prob = 0;
    {
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

        const float2 input_uv = get_uv(px + frame_constants.view_constants.sample_offset_pixels, input_tex_size);
        //const float2 input_uv = get_uv(px * 2 + 1 + frame_constants.view_constants.sample_offset_pixels, output_tex_size);
        /*float4 closest_history = image_sample_catmull_rom(
            TextureImage::from_parts(history_tex, output_tex_size.xy),
            input_uv,
            HistoryRemap::create()
        );*/
        const float4 closest_history = HistoryRemap::create().remap(history_tex.SampleLevel(sampler_llc, input_uv, 0));
        const float4 closest_meta = meta_history_tex.SampleLevel(sampler_lnc, input_uv, 0);
        const float2 closest_vel = velocity_history_tex.SampleLevel(sampler_lnc, input_uv, 0).xy * frame_constants.delta_time_seconds;
        const float closest_var = max(0.01, closest_meta.x);

        #if 1
        {
            int k = 0;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    const float3 s = rgb_to_ycbcr(decode_rgb(input_tex[px + int2(x, y)].rgb));
                    const float3 idiff = s - closest_history.rgb;

                    const float2 vel = reprojection_tex[px + int2(x, y)].xy;
                    const float vdiff = length(vel - closest_vel);

                    const float prob = exp2(-30 * length(idiff * idiff) - 0 * length(vdiff * vdiff));

                    //float prob = exp2(-0.1 * length(idiff * idiff / max(1e-3, closest_var)));
                    input_prob = max(input_prob, prob);
                }
            }
        }
        #else
            float3 idiff = input - closest_history.rgb;
        //input_prob = exp2(-0.1 * length(idiff * idiff / max(0.0001, ivar)));
        //input_prob = exp2(-0.1 * length(idiff * idiff / max(0.0001, smooth_var)));
            //input_prob = exp2(-8 * length(idiff * idiff));
            input_prob = exp2(-0.1 * length(idiff * idiff / closest_var));
        #endif
    }

    output_tex[px] = input_prob;
}
