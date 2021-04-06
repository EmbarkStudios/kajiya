#include "inc/frame_constants.hlsl"
#include "inc/pack_unpack.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] RWTexture2D<float4> output_tex;
[[vk::binding(2)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

[numthreads(8, 8, 1)]
void main(in int2 px : SV_DispatchThreadID) {
    // TODO: use gbuffer unpacking
    int2 src_px = int2(((px + 0.25) / output_tex_size.xy) * input_tex_size.xy);
    float3 normal = unpack_normal_11_10_11_no_normalize(input_tex[src_px].y);
    float3 normal_vs = normalize(mul(frame_constants.view_constants.world_to_view, float4(normal, 0)).xyz);
	output_tex[px] = float4(normal_vs, 1);
}
