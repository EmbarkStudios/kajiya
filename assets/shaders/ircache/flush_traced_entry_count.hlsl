#include "ircache_constants.hlsl"

[[vk::binding(0)]] RWByteAddressBuffer ircache_meta_buf;

[numthreads(1, 1, 1)]
void main() {
    const uint entry_count = ircache_meta_buf.Load(IRCACHE_META_TRACING_ENTRY_COUNT);
    ircache_meta_buf.Store(IRCACHE_META_TRACED_ENTRY_COUNT, entry_count);
}
