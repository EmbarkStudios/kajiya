#include "surfel_constants.hlsl"

[[vk::binding(0)]] ByteAddressBuffer surfel_meta_buf;
[[vk::binding(1)]] RWByteAddressBuffer dispatch_args;

[numthreads(1, 1, 1)]
void main() {
    const uint surfel_count = surfel_meta_buf.Load(SURFEL_META_SURFEL_COUNT);
    dispatch_args.Store4(0, uint4(surfel_count, 1, 1, 0));
}