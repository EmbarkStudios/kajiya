#include "ircache_constants.hlsl"

[[vk::binding(0)]] RWByteAddressBuffer ircache_meta_buf;
[[vk::binding(1)]] RWByteAddressBuffer dispatch_args;

[numthreads(1, 1, 1)]
void main() {
    const uint entry_count = ircache_meta_buf.Load(IRCACHE_META_ENTRY_COUNT);
    const uint alloc_count = ircache_meta_buf.Load(IRCACHE_META_ALLOC_COUNT);

    ircache_meta_buf.Store(IRCACHE_META_TRACING_ALLOC_COUNT, alloc_count);

    // Reset, sum up irradiance
    dispatch_args.Store4(16 * 2, uint4((alloc_count + 63) / 64, 1, 1, 0));

    uint main_rt_samples = alloc_count * IRCACHE_SAMPLES_PER_FRAME;
    uint accessibility_rt_samples = alloc_count * IRCACHE_OCTA_DIMS2;
    uint validity_rt_samples = alloc_count * IRCACHE_VALIDATION_SAMPLES_PER_FRAME;

    // AMD ray-tracing bug workaround; indirect RT seems to be tracing with the same
    // ray count for multiple dispatches (???)
    // Search for c804a814-fdc8-4843-b2c8-9d0674c10a6f for other occurences.
    #if 1
        const uint max_rt_samples =
            max(main_rt_samples, max(accessibility_rt_samples, validity_rt_samples));

        main_rt_samples = max_rt_samples;
        accessibility_rt_samples = max_rt_samples;
        validity_rt_samples = max_rt_samples;
    #endif

    // Main ray tracing
    dispatch_args.Store4(16 * 0, uint4(main_rt_samples, 1, 1, 0));

    // Accessibility tracing
    dispatch_args.Store4(16 * 1, uint4(accessibility_rt_samples, 1, 1, 0));

    // Validity check
    dispatch_args.Store4(16 * 3, uint4(validity_rt_samples, 1, 1, 0));
}
