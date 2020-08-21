#include "sdf_consts.hlsl"

struct BrickInstance {
    float3 position;
    float pad;
};

[[vk::binding(0)]] Texture3D<float> sdf_tex;
[[vk::binding(1)]] RWByteAddressBuffer bricks_meta;
[[vk::binding(2)]] RWStructuredBuffer<BrickInstance> bricks_buffer;

[numthreads(4, 4, 4)]
void main(in uint3 pix: SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    float mind = 1.0;
    for (uint z = 0; z < 2; ++z) {
        for (uint y = 0; y < 2; ++y) {
            for (uint x = 0; x < 2; ++x) {
                mind = min(mind, sdf_tex[pix * 2 + uint3(x, y, z)]);
            }
        }
    }

    mind = WaveActiveMin(mind);

    if (0 == idx_within_group) {
        if (mind < 0.0) {
            uint brick_addr = 0;
            bricks_meta.InterlockedAdd(0, 1, brick_addr);

            BrickInstance binst = (BrickInstance)0;
            binst.position = float3(pix);   // TODO

            bricks_buffer[brick_addr] = binst;
        }
    }
}
