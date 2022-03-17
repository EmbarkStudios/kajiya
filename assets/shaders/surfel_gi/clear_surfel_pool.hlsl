[[vk::binding(0)]] RWStructuredBuffer<uint> surf_rcache_pool_buf;

[numthreads(64, 1, 1)]
void main(uint idx: SV_DispatchThreadID) {
    surf_rcache_pool_buf[idx] = idx;
}
