#include "sdf_consts.hlsl"

RWTexture3D<float> outputTex;

[numthreads(4, 4, 4)]
void main(in uint3 pix : SV_DispatchThreadID) {
    outputTex[pix] = SDF_EMPTY_DIST;
}
