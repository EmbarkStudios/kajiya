#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/sh.hlsl"
#include "ircache_constants.hlsl"

[[vk::binding(0)]] StructuredBuffer<uint> ircache_life_buf;
[[vk::binding(1)]] ByteAddressBuffer ircache_meta_buf;
[[vk::binding(2)]] StructuredBuffer<float4> ircache_irradiance_buf;
[[vk::binding(3)]] RWStructuredBuffer<float4> ircache_aux_buf;
[[vk::binding(4)]] StructuredBuffer<uint> ircache_entry_indirection_buf;

[numthreads(64, 1, 1)]
void main(uint dispatch_idx: SV_DispatchThreadID) {
    if (IRCACHE_FREEZE) {
        return;
    }

    const uint total_alloc_count = ircache_meta_buf.Load(IRCACHE_META_ALLOC_COUNT);
    if (dispatch_idx >= total_alloc_count) {
        return;
    }

    const uint entry_idx = ircache_entry_indirection_buf[dispatch_idx];

    const uint total_entry_count = ircache_meta_buf.Load(IRCACHE_META_TRACING_ENTRY_COUNT);
    const uint life = ircache_life_buf[entry_idx];

    if (entry_idx >= total_entry_count || !is_ircache_entry_life_valid(life)) {
        return;
    }

    const bool should_reset = all(0.0 == ircache_irradiance_buf[entry_idx * IRCACHE_IRRADIANCE_STRIDE]);

    if (should_reset) {
        for (uint i = 0; i < IRCACHE_AUX_STRIDE; ++i) {
            ircache_aux_buf[entry_idx * IRCACHE_AUX_STRIDE + i] = 0.0.xxxx;
        }
    }
}
