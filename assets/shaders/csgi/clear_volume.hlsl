[[vk::binding(0)]] RWTexture3D<float4> output_tex;

[numthreads(4, 4, 4)]
void main(in uint3 pix : SV_DispatchThreadID) {
    output_tex[pix] = float4(0, 0, 0, 0);

    /*
        pix *= 2;

    [unroll]
    for (uint z = 0; z < 2; ++z) {
        [unroll]
        for (uint y = 0; y < 2; ++y) {
            [unroll]
            for (uint x = 0; x < 2; ++x) {
                output_tex[pix + uint3(x, y, z)] = float4(0, 0, 0, 0);
            }
        }
    }
*/
}
