#include "../inc/frame_constants.hlsl"
#include "../inc/color/srgb.hlsl"

#include "luminance_histogram_common.hlsl"

[[vk::binding(0)]] Texture2D<float3> input_tex;
[[vk::binding(1)]] RWStructuredBuffer<uint> output_buffer;
[[vk::binding(2)]] cbuffer _ {
    uint2 input_extent;
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    if (any(px >= input_extent)) {
        return;
    }

    float log_lum = log2(max(1e-20, sRGB_to_luminance(input_tex[px]) / frame_constants.pre_exposure));

    const float t = saturate((log_lum - LUMINANCE_HISTOGRAM_MIN_LOG2) / (LUMINANCE_HISTOGRAM_MAX_LOG2 - LUMINANCE_HISTOGRAM_MIN_LOG2));
    const uint bin = min(uint(t * 256), 255);

    const float2 uv = float2(px + 0.5) / input_extent;
    const float infl = exp(-8 * pow(length(uv - 0.5), 2));
    const uint quantized_infl = uint(infl * 256.0);

    InterlockedAdd(output_buffer[bin], quantized_infl);
}
