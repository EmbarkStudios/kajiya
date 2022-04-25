[[vk::binding(0)]] RWTexture1D<uint> output_tex;

[numthreads(256, 1, 1)]
void main(uint bin: SV_DispatchThreadID) {
    output_tex[bin] = 0;
}
