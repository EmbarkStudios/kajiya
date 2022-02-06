[[vk::binding(0)]] RWStructuredBuffer<uint> surfel_pool_buf;

[numthreads(64, 1, 1)]
void main(uint idx: SV_DispatchThreadID) {
    surfel_pool_buf[idx] = idx;
}
