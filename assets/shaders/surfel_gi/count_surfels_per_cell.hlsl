#include "../inc/mesh.hlsl" // for VertexPacked

[[vk::binding(0)]] ByteAddressBuffer surfel_meta_buf;
[[vk::binding(1)]] ByteAddressBuffer surfel_hash_key_buf;
[[vk::binding(2)]] ByteAddressBuffer surfel_hash_value_buf;
[[vk::binding(3)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(4)]] RWByteAddressBuffer cell_index_offset_buf;

#include "surfel_grid_hash.hlsl"
#include "surfel_binning_shared.hlsl"

[numthreads(64, 1, 1)]
void main(uint surfel_idx: SV_DispatchThreadID) {
    const uint total_surfel_count = surfel_meta_buf.Load(1 * sizeof(uint));
    if (surfel_idx >= total_surfel_count) {
        return;
    }

    const Vertex surfel = unpack_vertex(surfel_spatial_buf[surfel_idx]);

    int3 box_min;
    int3 box_max;
    get_surfel_grid_box_min_max(surfel, box_min, box_max);

    for (int z = box_min.z; z <= box_max.z; ++z) {
        for (int y = box_min.y; y <= box_max.y; ++y) {
            for (int x = box_min.x; x <= box_max.x; ++x) {
                if (!surfel_intersects_grid_coord(surfel, int3(x, y, z))) {
                    continue;
                }

                const SurfelGridHashEntry entry = surfel_hash_lookup_by_grid_coord(int3(x, y, z));

                if (entry.found) {
                    const uint cell_idx = surfel_hash_value_buf.Load(sizeof(uint) * entry.idx);
                    cell_index_offset_buf.InterlockedAdd(sizeof(uint) * cell_idx, 1);
                }
            }
        }
    }
}