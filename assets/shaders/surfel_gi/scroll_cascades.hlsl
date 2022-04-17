#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"

[[vk::binding(0)]] ByteAddressBuffer surf_rcache_grid_meta_buf;
[[vk::binding(1)]] RWByteAddressBuffer surf_rcache_grid_meta_buf2;
[[vk::binding(2)]] RWStructuredBuffer<uint> surf_rcache_entry_cell_buf;
[[vk::binding(3)]] RWStructuredBuffer<float4> surf_rcache_irradiance_buf;
[[vk::binding(4)]] RWStructuredBuffer<uint> surf_rcache_life_buf;
[[vk::binding(5)]] RWStructuredBuffer<uint> surf_rcache_pool_buf;
[[vk::binding(6)]] RWByteAddressBuffer surf_rcache_meta_buf;

#include "surfel_constants.hlsl"
#include "surfel_grid_hash.hlsl"


void deallocate_cell(uint cell_idx) {
    const uint2 meta = surf_rcache_grid_meta_buf.Load2(sizeof(uint2) * cell_idx);

    if (meta.y & SURF_RCACHE_ENTRY_META_OCCUPIED) {
        // Clear the just-nuked entry
        const uint entry_idx = meta.x;

        surf_rcache_life_buf[entry_idx] = SURFEL_LIFE_RECYCLED;

        for (uint i = 0; i < SURF_RCACHE_IRRADIANCE_STRIDE; ++i) {
            surf_rcache_irradiance_buf[entry_idx * SURF_RCACHE_IRRADIANCE_STRIDE + i] = 0.0.xxxx;
        }

        uint surfel_alloc_count = 0;
        surf_rcache_meta_buf.InterlockedAdd(SURFEL_META_ALLOC_COUNT, -1, surfel_alloc_count);
        surf_rcache_pool_buf[surfel_alloc_count - 1] = entry_idx;
    }
}

// Note: group dims must be divisors of RCACHE_CASCADE_SIZE
[numthreads(32, 1, 1)]
void main(uint3 dispatch_thread_id: SV_DispatchThreadID) {
    const uint3 dst_vx = uint3(dispatch_thread_id.xy, dispatch_thread_id.z % RCACHE_CASCADE_SIZE);
    const uint cascade = dispatch_thread_id.z / RCACHE_CASCADE_SIZE;

    const uint dst_cell_idx = RcacheCoord::from_coord_cascade(dst_vx, cascade).cell_idx();

    const int3 scroll_by =
        SURF_RCACHE_FREEZE
        ? (0).xxx
        : frame_constants.rcache_cascades[cascade].voxels_scrolled_this_frame.xyz;

    if (!all(uint3(dst_vx - scroll_by) < RCACHE_CASCADE_SIZE)) {
        // If this entry is about to get overwritten, deallocate it.
        deallocate_cell(dst_cell_idx);
    }

    const uint3 src_vx = dst_vx + scroll_by;

    if (all(src_vx < RCACHE_CASCADE_SIZE)) {
        const uint src_cell_idx = RcacheCoord::from_coord_cascade(src_vx, cascade).cell_idx();

        const uint2 cell_meta = surf_rcache_grid_meta_buf.Load2(sizeof(uint2) * src_cell_idx);
        surf_rcache_grid_meta_buf2.Store2(sizeof(uint2) * dst_cell_idx, cell_meta);

        // Update the cell idx in the `surf_rcache_entry_cell_buf`
        if (cell_meta.y & SURF_RCACHE_ENTRY_META_OCCUPIED) {
            const uint entry_idx = cell_meta.x;
            surf_rcache_entry_cell_buf[entry_idx] = dst_cell_idx;
        }
    } else {
        surf_rcache_grid_meta_buf2.Store2(sizeof(uint2) * dst_cell_idx, (0).xx);
    }
}
