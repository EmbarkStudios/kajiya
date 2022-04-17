#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked

[[vk::binding(0)]] RWByteAddressBuffer surf_rcache_meta_buf;
[[vk::binding(1)]] RWByteAddressBuffer surf_rcache_grid_meta_buf;
[[vk::binding(2)]] RWStructuredBuffer<uint> surf_rcache_entry_cell_buf;
[[vk::binding(3)]] RWStructuredBuffer<uint> surf_rcache_life_buf;
[[vk::binding(4)]] RWStructuredBuffer<uint> surf_rcache_pool_buf;
[[vk::binding(5)]] RWStructuredBuffer<VertexPacked> surf_rcache_spatial_buf;
[[vk::binding(6)]] RWStructuredBuffer<VertexPacked> surf_rcache_reposition_proposal_buf;
[[vk::binding(7)]] RWStructuredBuffer<uint> surf_rcache_reposition_proposal_count_buf;
[[vk::binding(8)]] RWStructuredBuffer<float4> surf_rcache_irradiance_buf;

#include "surfel_constants.hlsl"

void age_surfel(uint entry_idx) {
    const uint prev_age = surf_rcache_life_buf[entry_idx];
    const uint new_age = prev_age + 1;

    if (is_surfel_life_valid(new_age)) {
        surf_rcache_life_buf[entry_idx] = new_age;
    } else {
        surf_rcache_life_buf[entry_idx] = SURFEL_LIFE_RECYCLED;
        // onoz, we killed it!
        // deallocate.

        for (uint i = 0; i < SURF_RCACHE_IRRADIANCE_STRIDE; ++i) {
            surf_rcache_irradiance_buf[entry_idx * SURF_RCACHE_IRRADIANCE_STRIDE + i] = 0.0.xxxx;
        }

        uint surfel_alloc_count = 0;
        surf_rcache_meta_buf.InterlockedAdd(SURFEL_META_ALLOC_COUNT, -1, surfel_alloc_count);
        surf_rcache_pool_buf[surfel_alloc_count - 1] = entry_idx;

        // TODO: just `Store` it.
        const uint cell_idx = surf_rcache_entry_cell_buf[entry_idx];
        surf_rcache_grid_meta_buf.InterlockedAnd(sizeof(uint2) * cell_idx + sizeof(uint), ~SURF_RCACHE_ENTRY_META_OCCUPIED);
    }
}

[numthreads(64, 1, 1)]
void main(uint entry_idx: SV_DispatchThreadID) {
    if (SURF_RCACHE_FREEZE) {
        return;
    }

    const uint total_surfel_count = surf_rcache_meta_buf.Load(SURFEL_META_ENTRY_COUNT);
    
    if (entry_idx < total_surfel_count) {
        if (surfel_life_needs_aging(surf_rcache_life_buf[entry_idx])) {
            age_surfel(entry_idx);
        }

        #if 1
            #if 0
                uint rng = hash2(uint2(entry_idx, frame_constants.frame_index));
                const float dart = uint_to_u01_float(hash1_mut(rng));
                const float prob = 0.02;

                if (dart <= prob)
            #endif
            {

                // Flush the reposition proposal
                VertexPacked proposal = surf_rcache_reposition_proposal_buf[entry_idx];
                surf_rcache_spatial_buf[entry_idx] = proposal;
            }
        #endif

        surf_rcache_reposition_proposal_count_buf[entry_idx] = 0;
    } else {
        VertexPacked invalid;
        invalid.data0 = asfloat(0);
        surf_rcache_spatial_buf[entry_idx] = invalid;
    }
}
