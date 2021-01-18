#include "inc/tonemap.hlsl"

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

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float3 col = fetch_color(px);

    // TODO: move to its own pass
#if 1
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

    //col *= 2;
    //col -= 0.49;
    //col *= 10;
    col = neutral_tonemap(col);
    //col = 1-exp(-col);
    
    output_tex[px] = float4(col, 1.0);
}
