#include "../inc/mesh.hlsl" // for VertexPacked

[[vk::binding(0)]] ByteAddressBuffer surfel_meta_buf;
[[vk::binding(1)]] ByteAddressBuffer surfel_hash_key_buf;
[[vk::binding(2)]] ByteAddressBuffer surfel_hash_value_buf;
[[vk::binding(3)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(4)]] RWByteAddressBuffer cell_index_offset_buf;

#include "surfel_grid_hash.hlsl"

static const float SURFEL_RADIUS = 0.5;

[numthreads(64, 1, 1)]
void main(uint thread_index: SV_DispatchThreadID) {
    const Vertex surfel = unpack_vertex(surfel_spatial_buf[thread_index]);

    const float3 box_min_pos = surfel.position - SURFEL_RADIUS;
    const float3 box_max_pos = surfel.position + SURFEL_RADIUS;

    const int3 box_min = surfel_pos_to_grid_coord(box_min_pos);
    const int3 box_max = surfel_pos_to_grid_coord(box_max_pos);

    for (int z = box_min.z; z <= box_max.z; ++z) {
        for (int y = box_min.y; y <= box_max.y; ++y) {
            for (int x = box_min.x; x <= box_max.x; ++x) {
                const SurfelGridHashEntry entry = surfel_hash_lookup_by_grid_coord(int3(x, y, z));

                if (entry.found) {
                    const uint cell_idx = surfel_hash_value_buf.Load(sizeof(uint) * entry.idx);
                    cell_index_offset_buf.InterlockedAdd(sizeof(uint) * cell_idx, 1);
                }
            }
        }
    }
}