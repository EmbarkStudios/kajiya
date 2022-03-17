#include "surfel_constants.hlsl"

[[vk::binding(0)]] ByteAddressBuffer surf_rcache_meta_buf;
[[vk::binding(1)]] RWByteAddressBuffer dispatch_args;

[numthreads(1, 1, 1)]
void main() {
    // Aging ags
    {
        const uint surfel_count = surf_rcache_meta_buf.Load(SURFEL_META_ENTRY_COUNT);

        static const uint threads_per_group = 64;
        static const uint entries_per_thread = 1;
        static const uint divisor = threads_per_group * entries_per_thread;

        dispatch_args.Store4(0 * sizeof(uint4), uint4((surfel_count + divisor - 1) / divisor, 1, 1, 0));
    }
}