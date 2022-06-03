#include "../inc/frame_constants.hlsl"
#include "ircache_constants.hlsl"

[[vk::binding(0)]] RWByteAddressBuffer ircache_meta_buf;
[[vk::binding(1)]] RWStructuredBuffer<uint> ircache_life_buf;
[[vk::binding(2)]] StructuredBuffer<uint> entry_occupancy_buf;
[[vk::binding(3)]] RWStructuredBuffer<uint> ircache_entry_indirection_buf;

[numthreads(64, 1, 1)]
void main(uint entry_idx: SV_DispatchThreadID) {
    const uint total_entry_count = ircache_meta_buf.Load(IRCACHE_META_ENTRY_COUNT);

    const uint life = ircache_life_buf[entry_idx];
    if (entry_idx < total_entry_count && is_ircache_entry_life_valid(life)) {
        ircache_entry_indirection_buf[entry_occupancy_buf[entry_idx]] = entry_idx;
    }
}
