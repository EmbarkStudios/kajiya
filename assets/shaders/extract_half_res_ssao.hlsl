#include "inc/frame_constants.hlsl"
#include "inc/pack_unpack.hlsl"

[[vk::binding(0)]] Texture2D<float> input_tex;
[[vk::binding(1)]] RWTexture2D<float> output_tex;
[[vk::binding(2)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

[numthreads(8, 8, 1)]
void main(in int2 px : SV_DispatchThreadID) {
    uint2 hi_px_subpixels[4] = {
        uint2(0, 0),
        uint2(1, 1),
        uint2(1, 0),
        uint2(0, 1),
    };
    const int2 src_px = px * 2 + hi_px_subpixels[frame_constants.frame_index & 3];

	output_tex[px] = input_tex[src_px];
}
