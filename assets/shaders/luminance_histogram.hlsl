#include "inc/frame_constants.hlsl"
#include "inc/color/srgb.hlsl"

#include "luminance_histogram_common.hlsl"

[[vk::binding(0)]] Texture2D<float3> input_tex;
[[vk::binding(1)]] RWTexture1D<uint> output_tex;
[[vk::binding(2)]] cbuffer _ {
    uint2 input_extent;
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    if (any(px >= input_extent)) {
        return;
    }

    float log_lum = log2(max(1e-20, sRGB_to_luminance(input_tex[px]) / frame_constants.pre_exposure));

    const float t = saturate((log_lum - LUMINANCE_HISTOGRAM_MIN_LOG) / (LUMINANCE_HISTOGRAM_MAX_LOG - LUMINANCE_HISTOGRAM_MIN_LOG));
    const uint bin = min(uint(t * 256), 255);

    InterlockedAdd(output_tex[bin], 1);
}
