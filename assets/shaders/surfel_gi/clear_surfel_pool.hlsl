#include "surfel_constants.hlsl"

[[vk::binding(0)]] RWStructuredBuffer<uint> surf_rcache_pool_buf;
[[vk::binding(1)]] RWStructuredBuffer<uint> surf_rcache_life_buf;

[numthreads(64, 1, 1)]
void main(uint idx: SV_DispatchThreadID) {
    surf_rcache_pool_buf[idx] = idx;
    //surf_rcache_life_buf[idx] = SURFEL_LIFE_RECYCLED;
}
