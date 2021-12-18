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

float3 filter_input(uint2 px, float center_depth, float luma_cutoff, float depth_scale) {
    const float ang_offset = uint_to_u01_float(hash3(
        uint3(px, frame_constants.frame_index)
    ));

    float3 min_s = 10000000.0;

    float3 iex = 0;
    float iwsum = 0;

    const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const int2 spx_offset = int2(x, y);
            const float r_w = exp(-(0.8 / (k * k)) * dot(spx_offset, spx_offset));

            const int2 spx = int2(px) + spx_offset;
            float3 s = rgb_to_ycbcr(decode_rgb(input_tex[spx].rgb));

            const float depth = depth_tex[spx];
            float w = 1;
            w *= exp2(-min(16, depth_scale * inverse_depth_relative_diff(center_depth, depth)));
            w *= r_w;
            w *= pow(saturate(luma_cutoff / s.x), 8);

            if (s.x < min_s.x) {
                min_s = s;
            }

            iwsum += w;
            iex += s * w;
        }
    }

    return iex / iwsum;
    //return min_s;
}

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    const float center_depth = depth_tex[px];
    float filtered_luma = filter_input(px, center_depth, 1e10, 200).x;
    output_tex[px] = filter_input(px, center_depth, filtered_luma * 1.001, 200);
}
