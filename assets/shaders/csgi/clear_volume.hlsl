[[vk::binding(0)]] RWTexture3D<float4> output_tex;

[numthreads(4, 4, 4)]
void main(in uint3 pix : SV_DispatchThreadID) {
    output_tex[pix] = float4(0, 0, 0, 0);
}
