#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked

[[vk::binding(0)]] RWByteAddressBuffer ircache_meta_buf;
[[vk::binding(1)]] RWByteAddressBuffer ircache_grid_meta_buf;
[[vk::binding(2)]] RWStructuredBuffer<uint> ircache_entry_cell_buf;
[[vk::binding(3)]] RWStructuredBuffer<uint> ircache_life_buf;
[[vk::binding(4)]] RWStructuredBuffer<uint> ircache_pool_buf;
[[vk::binding(5)]] RWStructuredBuffer<VertexPacked> ircache_spatial_buf;
[[vk::binding(6)]] RWStructuredBuffer<VertexPacked> ircache_reposition_proposal_buf;
[[vk::binding(7)]] RWStructuredBuffer<uint> ircache_reposition_proposal_count_buf;
[[vk::binding(8)]] RWStructuredBuffer<float4> ircache_irradiance_buf;
[[vk::binding(9)]] RWStructuredBuffer<uint> entry_occupancy_buf;

#include "ircache_constants.hlsl"

void age_ircache_entry(uint entry_idx) {
    const uint prev_age = ircache_life_buf[entry_idx];
    const uint new_age = prev_age + 1;

    if (is_ircache_entry_life_valid(new_age)) {
        ircache_life_buf[entry_idx] = new_age;

        // TODO: just `Store` it (AMD doesn't like it unless it's a byte address buffer)
        const uint cell_idx = ircache_entry_cell_buf[entry_idx];
        ircache_grid_meta_buf.InterlockedAnd(
            sizeof(uint2) * cell_idx + sizeof(uint),
            ~IRCACHE_ENTRY_META_JUST_ALLOCATED);
    } else {
        ircache_life_buf[entry_idx] = IRCACHE_ENTRY_LIFE_RECYCLED;
        // onoz, we killed it!
        // deallocate.

        for (uint i = 0; i < IRCACHE_IRRADIANCE_STRIDE; ++i) {
            ircache_irradiance_buf[entry_idx * IRCACHE_IRRADIANCE_STRIDE + i] = 0.0.xxxx;
        }

        uint entry_alloc_count = 0;
        ircache_meta_buf.InterlockedAdd(IRCACHE_META_ALLOC_COUNT, -1, entry_alloc_count);
        ircache_pool_buf[entry_alloc_count - 1] = entry_idx;

        // TODO: just `Store` it (AMD doesn't like it unless it's a byte address buffer)
        const uint cell_idx = ircache_entry_cell_buf[entry_idx];
        ircache_grid_meta_buf.InterlockedAnd(
            sizeof(uint2) * cell_idx + sizeof(uint),
            ~(IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_JUST_ALLOCATED));
    }
}

bool ircache_entry_life_needs_aging(uint life) {
    return life != IRCACHE_ENTRY_LIFE_RECYCLED;
}

[numthreads(64, 1, 1)]
void main(uint entry_idx: SV_DispatchThreadID) {
    const uint total_entry_count = ircache_meta_buf.Load(IRCACHE_META_ENTRY_COUNT);

    if (!IRCACHE_FREEZE) {
        if (entry_idx < total_entry_count) {
            const uint life = ircache_life_buf[entry_idx];

            if (ircache_entry_life_needs_aging(life)) {
                age_ircache_entry(entry_idx);
            }

            #if IRCACHE_USE_POSITION_VOTING
                #if 0
                    uint rng = hash2(uint2(entry_idx, frame_constants.frame_index));
                    const float dart = uint_to_u01_float(hash1_mut(rng));
                    const float prob = 0.02;

                    if (dart <= prob)
                #endif
                {

                    // Flush the reposition proposal
                    VertexPacked proposal = ircache_reposition_proposal_buf[entry_idx];
                    ircache_spatial_buf[entry_idx] = proposal;
                }
            #endif

            ircache_reposition_proposal_count_buf[entry_idx] = 0;
        } else {
            VertexPacked invalid;
            invalid.data0 = asfloat(0);
            ircache_spatial_buf[entry_idx] = invalid;
        }
    }

    const uint life = ircache_life_buf[entry_idx];
    uint valid = entry_idx < total_entry_count && is_ircache_entry_life_valid(life);
    entry_occupancy_buf[entry_idx] = valid;
}
