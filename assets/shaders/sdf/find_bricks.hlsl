#include "sdf_consts.hlsl"

struct BrickInstance {
    float3 position;
    float half_extent;
};

[[vk::binding(0)]] Texture3D<float> sdf_tex;
[[vk::binding(1)]] RWByteAddressBuffer bricks_meta;
[[vk::binding(2)]] RWStructuredBuffer<BrickInstance> bricks_buffer;

groupshared uint group_any_brick_occupied;

[numthreads(4, 4, 4)]
void main(in uint3 pix: SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex, in uint3 brick_idx: SV_GroupID) {
    float mind = 1.0;
    for (uint z = 0; z < 2; ++z) {
        for (uint y = 0; y < 2; ++y) {
            for (uint x = 0; x < 2; ++x) {
                mind = min(mind, sdf_tex[pix * 2 + uint3(x, y, z)]);
            }
        }
    }

    if (0 == idx_within_group) {
        group_any_brick_occupied = false;
    }

    GroupMemoryBarrierWithGroupSync();

    bool occupied_brick = WaveActiveAnyTrue(mind < 0.0);

    if (WaveIsFirstLane()) {
        bool _orig;
        InterlockedOr(group_any_brick_occupied, uint(occupied_brick), _orig);
    }

    GroupMemoryBarrierWithGroupSync();    

    if (0 == idx_within_group && group_any_brick_occupied) {
        uint brick_addr = 0;

        // Add to the `instanceCount` field of `VkDrawIndirectCommand` stored in `bricks_meta`
        bricks_meta.InterlockedAdd(4, 1, brick_addr);

        float voxel_size = 2.0 * HSIZE / SDFRES;
        float brick_size = voxel_size * BRICKRES;
        float3 brick_center = (brick_idx - BRICK_GRID_RES * 0.5) * brick_size + brick_size * 0.5;

        // TODO: figure out the math
        BrickInstance binst;
        binst.position = brick_center;
        binst.half_extent = brick_size * 0.5;

        bricks_buffer[brick_addr] = binst;
    }
}
