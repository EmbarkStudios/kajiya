#include "sdf_consts.hlsl"

RWTexture3D<float> output_tex;

[numthreads(4, 4, 4)]
void main(in uint3 pix : SV_DispatchThreadID) {
    float3 ws_pos = ((float3(pix) / float3(SDFRES, SDFRES, SDFRES)) - 0.5) * 2.0 * HSIZE;

    output_tex[pix] = SDF_EMPTY_DIST;
    //output_tex[pix] = length(ws_pos) - 2.0;
}
