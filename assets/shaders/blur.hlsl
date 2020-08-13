Texture2D<float4> input_tex;
RWTexture2D<float4> output_tex;

[numthreads(8, 8, 1)]
void main(in uint2 pix : SV_DispatchThreadID) {
    float4 res = 0.0.xxxx;

    for (int i = -2; i <= 2; ++i) {
        res += input_tex[uint2(pix.x, pix.y + i * 4)];
    }

    res /= 5.0;

    output_tex[pix] = res;
}
