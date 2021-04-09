#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> direct_tex;
[[vk::binding(1)]] RWTexture3D<float> direct_opacity_tex;

#define USE_INDIRECT_CLAMP 1
#define INDIRECT_CLAMP_DIRECTIONAL 0
#define INDIRECT_CLAMP_AMOUNT 0.25

[numthreads(8, 8, 1)]
void main(uint3 vx: SV_DispatchThreadID) {
    float opacity = 0;

    [unroll]
    for (uint i = 0; i < 6; ++i) {
        opacity += direct_tex[vx + uint3(i * CSGI_VOLUME_DIMS, 0, 0)].a;
    }

    float light_mult = 1.0 - saturate(opacity * INDIRECT_CLAMP_AMOUNT);

    direct_opacity_tex[vx] = light_mult;
}
