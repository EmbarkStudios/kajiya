[[vk::binding(0)]] StructuredBuffer<uint> src_buffer;
[[vk::binding(1)]] RWStructuredBuffer<uint> dst_buffer;

[numthreads(64, 1, 1)]
void main(uint bin: SV_DispatchThreadID) {
    dst_buffer[bin] = src_buffer[bin];
}
