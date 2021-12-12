#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/math.hlsl"
#include "taa_common.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] RWTexture2D<float3> output_tex;

// Approximate Gaussian remap
// https://www.shadertoy.com/view/MlVSzw
float inv_error_function(float x, float truncation) {
    static const float ALPHA = 0.14;
    static const float INV_ALPHA = 1.0 / ALPHA;
    static const float K = 2.0 / (M_PI * ALPHA);

	float y = log(max(truncation, 1.0 - x*x));
	float z = K + 0.5 * y;
	return sqrt(max(0.0, sqrt(z*z - y * INV_ALPHA) - z)) * sign(x);
}

float remap_unorm_to_gaussian(float x, float truncation) {
	return inv_error_function(x * 2.0 - 1.0, truncation);
}

struct InputRemap {
    static InputRemap create() {
        InputRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return float4(rgb_to_ycbcr(decode_rgb(v.rgb)), 1);
    }
};

float3 filter_input(uint2 px, float luma_cutoff) {
    const float ang_offset = uint_to_u01_float(hash3(
        uint3(px, frame_constants.frame_index)
    ));

    float3 iex = 0;
    float iwsum = 0;
    {
    #if 1
        const int k = 2;
        for (int y = -k; y <= k; ++y)
        for (int x = -k; x <= k; ++x) {
            const int2 spx_offset = int2(x, y);
            const float r_w = exp(-(0.8 / (k * k)) * dot(spx_offset, spx_offset));
    #else
        const int k = 16;
        for (uint sample_i = 0; sample_i < k; ++sample_i) {
            float ang = (sample_i + ang_offset) * GOLDEN_ANGLE * M_TAU;
            float radius = 0.5 + pow(float(sample_i) / k, 0.5) * 3.5;
            const int2 spx_offset = float2(cos(ang), sin(ang)) * radius;
            const float r_w = 1;
    #endif
            const int2 spx = int2(px) + spx_offset;
            float3 s = rgb_to_ycbcr(input_tex[spx].rgb);

            float w = 1;
            w *= r_w;
            w *= pow(saturate(luma_cutoff / s.x), 8);
            //w *= saturate(luma_cutoff / s.x);

            iwsum += w;
            iex += s * w;
        }
    }

    return iex / iwsum;
}

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    #if 1
        float filtered_luma = filter_input(px, 1e10).x;
        output_tex[px] = filter_input(px, filtered_luma * 1.001);
    #elif 0
        float3 filtered = filter_input(px, 1e10);
        float3 raw = rgb_to_ycbcr(input_tex[px].rgb);
        output_tex[px] = raw * min(1.0, filtered.x / max(1e-5, raw.x));
    #else
        output_tex[px] = rgb_to_ycbcr(input_tex[px].rgb);
    #endif
}
