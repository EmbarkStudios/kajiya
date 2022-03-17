#include "surfel_constants.hlsl"

[[vk::binding(0)]] ByteAddressBuffer surf_rcache_meta_buf;
[[vk::binding(1)]] RWByteAddressBuffer dispatch_args;

[numthreads(1, 1, 1)]
void main() {
    const uint surfel_count = surf_rcache_meta_buf.Load(SURFEL_META_ENTRY_COUNT);
    dispatch_args.Store4(0, uint4(surfel_count, 1, 1, 0));
}