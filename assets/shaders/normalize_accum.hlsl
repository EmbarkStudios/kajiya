#include "inc/frame_constants.hlsl"
#include "inc/tonemap.hlsl"
#include "inc/bindless_textures.hlsl"

#define USE_TONEMAP 1
#define USE_DITHER 1

Texture2D<float4> input_tex;
RWTexture2D<float4> output_tex;

float3 fetch_color(uint2 px) {
    float4 res = input_tex[px];
    return res.xyz / res.w;
}

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

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float3 col = fetch_color(px);

    // TODO: move to its own pass
#if 0
    static const float sharpen_amount = 0.4;

	float neighbors = 0;
	float wt_sum = 0;

	const int2 dim_offsets[] = { int2(1, 0), int2(0, 1) };

	float center = sharpen_remap(calculate_luma(col.rgb));
    float2 wts;

	for (int dim = 0; dim < 2; ++dim) {
		int2 n0coord = px + dim_offsets[dim];
		int2 n1coord = px - dim_offsets[dim];

		float n0 = sharpen_remap(calculate_luma(fetch_color(n0coord).rgb));
		float n1 = sharpen_remap(calculate_luma(fetch_color(n1coord).rgb));
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

#if USE_TONEMAP
    //col *= 2;
    //col -= 0.47;
    col *= 8;
    col = neutral_tonemap(col);
    //col = 1-exp(-col);

    col = lerp(calculate_luma(col), col, 1.05);
    col = pow(col, 1.02);
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

    output_tex[px] = float4(col, 1.0);
}
