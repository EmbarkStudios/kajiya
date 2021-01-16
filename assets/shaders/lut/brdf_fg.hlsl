[[vk::binding(0)]] RWTexture2D<float4> output_tex;

[numthreads(8, 8, 1)]
void main(in uint2 pix : SV_DispatchThreadID) {
    output_tex[pix] = float4(0, 1, 0, 1);
}
