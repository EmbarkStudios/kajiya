[[vk::binding(0)]] Texture2D<float> input_tex;
[[vk::binding(1)]] RWTexture2D<float> output_tex;


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    output_tex[px] = input_tex[px];
}
