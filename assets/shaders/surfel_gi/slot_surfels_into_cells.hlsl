#include "../inc/mesh.hlsl" // for VertexPacked

[[vk::binding(0)]] ByteAddressBuffer surfel_meta_buf;
[[vk::binding(1)]] ByteAddressBuffer surfel_hash_key_buf;
[[vk::binding(2)]] ByteAddressBuffer surfel_hash_value_buf;
[[vk::binding(3)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(4)]] RWByteAddressBuffer cell_index_offset_buf;
[[vk::binding(5)]] RWByteAddressBuffer surfel_index_buf;

#include "surfel_grid_hash.hlsl"
#include "surfel_binning_shared.hlsl"

[numthreads(64, 1, 1)]
void main(uint surfel_idx: SV_DispatchThreadID) {
    const uint total_surfel_count = surfel_meta_buf.Load(SURFEL_META_SURFEL_COUNT);
    if (surfel_idx >= total_surfel_count) {
        return;
    }

    const Vertex surfel = unpack_vertex(surfel_spatial_buf[surfel_idx]);

    SurfelGridMinMax box = get_surfel_grid_box_min_max(surfel);
    for (uint ci = 0; ci < box.cascade_count; ++ci) {
        for (uint z = box.c4_min[ci].z; z <= box.c4_max[ci].z; ++z) {
            for (uint y = box.c4_min[ci].y; y <= box.c4_max[ci].y; ++y) {
                for (uint x = box.c4_min[ci].x; x <= box.c4_max[ci].x; ++x) {
                    const uint4 c4 = uint4(x, y, z, box.c4_min[ci].w);

                    if (!surfel_intersects_grid_coord(surfel, c4)) {
                        continue;
                    }

                    //const SurfelGridHashEntry entry = surfel_hash_lookup_by_grid_coord(int3(x, y, z));
                    const uint entry_idx = surfel_grid_c4_to_hash(c4);
     
                     //if (entry.found)
                     {
                        const uint cell_idx = surfel_hash_value_buf.Load(sizeof(uint) * entry_idx);

                        uint cell_index_loc_plus_one;
                        cell_index_offset_buf.InterlockedAdd(sizeof(uint) * cell_idx, -1, cell_index_loc_plus_one);

                        surfel_index_buf.Store(sizeof(uint) * (cell_index_loc_plus_one - 1), surfel_idx);
                    }
                }
            }
        }
    }
}