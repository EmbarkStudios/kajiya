#include "inc/samplers.hlsl"
#include "inc/uv.hlsl"
#include "inc/tonemap.hlsl"
#include "inc/frame_constants.hlsl"
#include "inc/bindless_textures.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
//[[vk::binding(1)]] Texture2D<float4> debug_input_tex;
[[vk::binding(1)]] Texture2D<float4> blur_pyramid_tex;
[[vk::binding(2)]] Texture2D<float4> rev_blur_pyramid_tex;
//[[vk::binding(4)]] Texture2D<float2> filtered_luminance_tex;
[[vk::binding(3)]] RWTexture2D<float4> output_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 output_tex_size;
    uint blur_pyramid_mip_count;
    float ev_shift;
};

#define USE_TONEMAP 1
#define USE_TIGHT_BLUR 0
#define USE_DITHER 1
#define USE_SHARPEN 1

static const float sharpen_amount = 0.2;
static const float glare_amount = 0.07;
//static const float glare_amount = 0.0;

float sharpen_remap(float l) {
    return sqrt(l);
}

float sharpen_inv_remap(float l) {
    return l * l;
}

float triangle_remap(float n) {
    float origin = n * 2.0 - 1.0;
    float v = origin * rsqrt(abs(origin));
    v = max(-1.0, v);
    v -= sign(origin);
    return v;
}

float local_tmo_constrain(float x, float max_compression) {
    #define local_tmo_constrain_mode 2

    #if local_tmo_constrain_mode == 0
        return exp(tanh(log(x) / max_compression) * max_compression);
    #elif local_tmo_constrain_mode == 1

        x = log(x);
        float s = sign(x);
        x = sqrt(abs(x));
        x = tanh(x / max_compression) * max_compression;
        x = exp(x * x * s);

        return x;
    #elif local_tmo_constrain_mode == 2
        float k = 3.0 * max_compression;
        x = 1.0 / x;
        x = tonemap_curve(x / k) * k;
        x = 1.0 / x;
        x = tonemap_curve(x / k) * k;
        return x;
    #else
        return x;
    #endif
}


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);

#if 0
    output_tex[px] = input_tex[px];
    return;
#endif

    float3 glare = rev_blur_pyramid_tex.SampleLevel(sampler_lnc, uv, 0).rgb;
    float3 col = input_tex[px].rgb;

    // TODO: move to its own pass
#if USE_SHARPEN
	float neighbors = 0;
	float wt_sum = 0;

	const int2 dim_offsets[] = { int2(1, 0), int2(0, 1) };

	float center = sharpen_remap(calculate_luma(col.rgb));
    float2 wts;

	for (int dim = 0; dim < 2; ++dim) {
		int2 n0coord = px + dim_offsets[dim];
		int2 n1coord = px - dim_offsets[dim];

		float n0 = sharpen_remap(calculate_luma(input_tex[n0coord].rgb));
		float n1 = sharpen_remap(calculate_luma(input_tex[n1coord].rgb));
		float wt = max(0, 1.0 - 6.0 * (abs(center - n0) + abs(center - n1)));
        wt = min(wt, sharpen_amount * wt * 1.25);
        
		neighbors += n0 * wt;
		neighbors += n1 * wt;
		wt_sum += wt * 2;
	}

    float sharpened_luma = max(0, center * (wt_sum + 1) - neighbors);
    sharpened_luma = sharpen_inv_remap(sharpened_luma);

	col.rgb *= max(0.0, sharpened_luma / max(1e-5, calculate_luma(col.rgb)));
#endif

    col = lerp(col, glare, glare_amount);
    col = max(0.0, col);
    //col = col * (1.0 - debug_input_tex[px].a) + debug_input_tex[px].rgb;

    col *= exp2(ev_shift);
    
#if USE_TONEMAP

    /*float filtered_luminance = exp(filtered_luminance_tex[px].x);
    float filtered_luminance_high = filtered_luminance_tex[px].y;

    float avg_luminance = 0;
    for (float y = 0.05; y < 1.0; y += 0.1) {
        for (float x = 0.05; x < 1.0; x += 0.1) {
            avg_luminance += filtered_luminance_tex[int2(output_tex_size.xy * float2(x, y))].x;
        }
    }
    avg_luminance = exp(avg_luminance / (10 * 10));

    const float lum_scale = 0.4;*/
    #if 0
        float avg_mult = lum_scale * 0.333 / avg_luminance;
        float mult = lum_scale * 0.333 / filtered_luminance;
        float relative_mult = mult / avg_mult;
        float max_compression = 0.5;
        float relative_shift = 1.1;
        relative_mult = local_tmo_constrain(relative_mult / relative_shift, max_compression);
        float remapped_mult = relative_mult * avg_mult * relative_shift;
        remapped_mult = lerp(remapped_mult, avg_mult, 0.1);
        col *= remapped_mult;

        float lin_part = clamp(remapped_mult * (0.8 * filtered_luminance - 0.2 * filtered_luminance_high), 0.0, 0.5);
        col.rgb = neutral_tonemap(col.rgb, lin_part);
    #else
        //float filtered_luminance = filtered_luminance_tex[px].g;
        //col *= 0.333 / filtered_luminance;
        //col *= lum_scale * 0.333 / avg_luminance;

        //col /= 2;
        //col *= 2;
        //col *= 4;
        //col *= 16;

        col = neutral_tonemap(col);
        //col = 1-exp(-col);
    #endif

    // Boost saturation and contrast to compensate for the loss from glare
    col = saturate(lerp(calculate_luma(col), col, 1.05));
    col = pow(col, 1.03);
#endif

    // Dither
#if USE_DITHER
    const uint urand_idx = frame_constants.frame_index;
    // 256x256 blue noise
    float dither = triangle_remap(bindless_textures[BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0][
        (px + int2(urand_idx * 59, urand_idx * 37)) & 255
    ].x);

    col += dither / 256.0;
#endif

    //col = filtered_luminance;

    output_tex[px] = float4(col, 1);
}
