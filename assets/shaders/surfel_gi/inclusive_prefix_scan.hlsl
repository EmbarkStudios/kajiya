#define THREAD_GROUP_SIZE 512
#define SEGMENT_SIZE (THREAD_GROUP_SIZE * 2)

[[vk::binding(0)]] RWByteAddressBuffer inout_buf;

groupshared uint shared_data[SEGMENT_SIZE];

uint2 load_input2(uint idx, uint segment) {
    return inout_buf.Load2(sizeof(uint) * (idx + segment * SEGMENT_SIZE));
}

void store_output2(uint idx, uint segment, uint2 val) {
    inout_buf.Store2(sizeof(uint) * (idx + segment * SEGMENT_SIZE), val);
}

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint idx: SV_GroupThreadID, uint segment: SV_GroupID) {
    const uint STEP_COUNT = uint(log2(THREAD_GROUP_SIZE)) + 1;

    const uint2 input2 = load_input2(idx * 2, segment);
    shared_data[idx * 2] = input2.x;
    shared_data[idx * 2 + 1]	= input2.y;

    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint step = 0; step < STEP_COUNT; step++) {
        uint mask = (1 << step) - 1;
        uint rd_idx = ((idx >> step) << (step + 1)) + mask;
        uint wr_idx = rd_idx + 1 + (idx & mask);

        shared_data[wr_idx] += shared_data[rd_idx];

        GroupMemoryBarrierWithGroupSync();
    }

    store_output2(idx * 2, segment, uint2(shared_data[idx * 2], shared_data[idx * 2 + 1]));
}
