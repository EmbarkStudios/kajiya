#include "../inc/pack_unpack.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] RWTexture3D<float4> direct_tex;

[numthreads(8, 8, 1)]
void main(uint3 vx: SV_DispatchThreadID) {
    #if 1
        float4 v = direct_tex[vx];

        // Having this branch on makes the sweep passes slower o__O
        // Weird cache behavior?
        //if (any(v > 1e-5))
        {
            direct_tex[vx] = v * (1.0 - CSGI_ACCUM_HYSTERESIS);
        }
    #else
        direct_tex[vx] = 0.0;
    #endif
}
