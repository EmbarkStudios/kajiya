[[vk::binding(0)]] Texture2D<float3> input_tex;
[[vk::binding(1)]] RWTexture2D<float2> output_tex;

[numthreads(8, 8, 1)]
void main(int2 px: SV_DispatchThreadID) {
    // TODO
}
