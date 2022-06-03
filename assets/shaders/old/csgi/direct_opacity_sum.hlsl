#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> direct_tex;
[[vk::binding(1)]] RWTexture3D<float> direct_opacity_tex;

#define INDIRECT_CLAMP_DIRECTIONAL 0

// A bit leaky, but better on high spec surfaces and tight corners:
//#define INDIRECT_CLAMP_AMOUNT 0.25

// Almost leak free, but overdarkens
#define INDIRECT_CLAMP_AMOUNT 0.5

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
