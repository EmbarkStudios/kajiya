#include "inc/samplers.hlsl"
#include "inc/uv.hlsl"
#include "inc/tonemap.hlsl"
#include "inc/frame_constants.hlsl"
#include "inc/bindless_textures.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> blur_pyramid_tex;
[[vk::binding(2)]] Texture2D<float4> rev_blur_pyramid_tex;
[[vk::binding(3)]] RWTexture2D<float4> output_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 output_tex_size;
    uint blur_pyramid_mip_count;
};

#define USE_TONEMAP 1
#define USE_TIGHT_BLUR 1
#define USE_DITHER 1

static const float glare_amount = 0.07;

float triangle_remap(float n) {
    float origin = n * 2.0 - 1.0;
    float v = origin * rsqrt(abs(origin));
    v = max(-1.0, v);
    v -= sign(origin);
    return v;
}

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);

#if 0
    output_tex[px] = input_tex[px];
    return;
#endif

#if 0
    static const float glare_falloff = 1.25;
    float3 glare = 0; {
        float wt_sum = 1;
        for (uint mip = 0; mip < blur_pyramid_mip_count; ++mip) {
            float wt = 1.0 / pow(glare_falloff, mip);
            glare += blur_pyramid_tex.SampleLevel(sampler_lnc, uv, mip).rgb * wt;
            wt_sum += wt;
        }
        glare /= wt_sum;
    }
#else
    float3 glare = rev_blur_pyramid_tex.SampleLevel(sampler_lnc, uv, 0).rgb;
#endif

#if USE_TIGHT_BLUR
    float3 tight_glare = 0.0; {
        static const int k = 1;
        float wt_sum = 0;

        [unroll]
        for (int y = -k; y <= k; ++y) {
            [unroll]
            for (int x = -k; x <= k; ++x) {
                float wt = exp2(-6.0 * sqrt(float(x * x + y * y)));
                tight_glare += input_tex[px + uint2(x, y)].rgb * wt;
                wt_sum += wt;
            }
        }

        tight_glare /= wt_sum;
    }

    float3 col = lerp(tight_glare, glare, glare_amount);
#else
    float3 col = lerp(input_tex[px].rgb, glare, glare_amount);
#endif

    //col *= 0.3;
    //col *= 500;

    //col /= 2;
    //col *= 2;
    //col *= 4;
    //col *= 16;

#if USE_TONEMAP
    col = neutral_tonemap(col);
    //col = 1-exp(-col);

    //col = lerp(calculate_luma(col), col, 1.05);
    col = pow(col, 1.03);
#endif

    // Dither
#if USE_DITHER
    const uint urand_idx = frame_constants.frame_index;
    // 256x256 blue noise
    float dither = triangle_remap(bindless_textures[1][
        (px + int2(urand_idx * 59, urand_idx * 37)) & 255
    ].x);

    col += dither / 256.0;
#endif

    output_tex[px] = float4(col, 1);
}
