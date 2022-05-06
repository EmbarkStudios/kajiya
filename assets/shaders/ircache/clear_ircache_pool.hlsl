#include "ircache_constants.hlsl"

[[vk::binding(0)]] RWStructuredBuffer<uint> ircache_pool_buf;
[[vk::binding(1)]] RWStructuredBuffer<uint> ircache_life_buf;

[numthreads(64, 1, 1)]
void main(uint idx: SV_DispatchThreadID) {
    ircache_pool_buf[idx] = idx;
    ircache_life_buf[idx] = IRCACHE_ENTRY_LIFE_RECYCLED;
}
