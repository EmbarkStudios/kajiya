#include "../inc/mesh.hlsl" // for VertexPacked

[[vk::binding(0)]] RWByteAddressBuffer surfel_meta_buf;
[[vk::binding(1)]] RWStructuredBuffer<uint> surfel_life_buf;
[[vk::binding(2)]] RWStructuredBuffer<uint> surfel_pool_buf;
[[vk::binding(3)]] RWStructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(4)]] RWStructuredBuffer<VertexPacked> surfel_reposition_proposal_buf;

#include "surfel_constants.hlsl"

void age_surfel(uint surfel_idx) {
    const uint prev_age = surfel_life_buf[surfel_idx];
    const uint new_age = prev_age + 1;

    if (is_surfel_life_valid(new_age)) {
        surfel_life_buf[surfel_idx] = new_age;
    } else {
        surfel_life_buf[surfel_idx] = SURFEL_LIFE_RECYCLED;
        // onoz, we killed it!
        // deallocate.

        uint surfel_alloc_count = 0;
        surfel_meta_buf.InterlockedAdd(SURFEL_META_ALLOC_COUNT, -1, surfel_alloc_count);
        surfel_pool_buf[surfel_alloc_count - 1] = surfel_idx;
    }
}

[numthreads(64, 1, 1)]
void main(uint surfel_idx: SV_DispatchThreadID) {
    const uint total_surfel_count = surfel_meta_buf.Load(SURFEL_META_SURFEL_COUNT);
    
    if (surfel_idx < total_surfel_count) {
        if (surfel_life_needs_aging(surfel_life_buf[surfel_idx])) {
            age_surfel(surfel_idx);
        }

        #if 0
            VertexPacked proposal = surfel_reposition_proposal_buf[surfel_idx];
            if (asuint(proposal.data0.x) != 0) {
                surfel_spatial_buf[surfel_idx] = proposal;
                
                surfel_reposition_proposal_buf[surfel_idx].data0.x = asfloat(0);
            }
        #endif
    }
}
