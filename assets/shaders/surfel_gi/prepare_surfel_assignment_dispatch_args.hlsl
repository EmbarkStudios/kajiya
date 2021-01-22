[[vk::binding(0)]] ByteAddressBuffer surfel_meta_buf;
[[vk::binding(1)]] RWByteAddressBuffer dispatch_args;

[numthreads(1, 1, 1)]
void main() {
    const uint cell_count = surfel_meta_buf.Load(sizeof(uint) * 0);

    // Cell clearing args
    {
        static const uint threads_per_group = 64;
        static const uint entries_per_thread = 4;
        static const uint divisor = threads_per_group * entries_per_thread;

        dispatch_args.Store4(0 * sizeof(uint4), uint4((cell_count + divisor - 1) / divisor, 1, 1, 0));
    }

    // Surfel binning args
    {
        static const uint threads_per_group = 64;
        static const uint entries_per_thread = 1;
        static const uint divisor = threads_per_group * entries_per_thread;

        dispatch_args.Store4(1 * sizeof(uint4), uint4((cell_count + divisor - 1) / divisor, 1, 1, 0));
    }
}