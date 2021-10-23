#define THREAD_GROUP_SIZE 512
#define SEGMENT_SIZE (THREAD_GROUP_SIZE * 2)

[[vk::binding(0)]] ByteAddressBuffer input_buf;
[[vk::binding(1)]] RWByteAddressBuffer output_buf;

groupshared uint shared_data[SEGMENT_SIZE];

uint load_input(uint idx) {
    const uint segment_sum_idx = idx * SEGMENT_SIZE + SEGMENT_SIZE - 1;
    return input_buf.Load(sizeof(uint) * segment_sum_idx);
}

void store_output2(uint idx, uint2 val) {
    output_buf.Store2(sizeof(uint) * idx, val);
}

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint idx: SV_GroupThreadID, uint segment: SV_GroupID) {
    const uint STEP_COUNT = uint(log2(THREAD_GROUP_SIZE)) + 1;

    shared_data[idx * 2] = load_input(idx * 2);
    shared_data[idx * 2 + 1] = load_input(idx * 2 + 1);

    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint step = 0; step < STEP_COUNT; step++) {
        uint mask = (1 << step) - 1;
        uint rd_idx = ((idx >> step) << (step + 1)) + mask;
        uint wr_idx = rd_idx + 1 + (idx & mask);

        shared_data[wr_idx] += shared_data[rd_idx];

        GroupMemoryBarrierWithGroupSync();
    }

    store_output2(idx * 2, uint2(shared_data[idx * 2], shared_data[idx * 2 + 1]));
}
