[[vk::binding(0)]] RWByteAddressBuffer cell_index_offset_buf;

[numthreads(64, 1, 1)]
void main(uint thread_index: SV_DispatchThreadID) {
    cell_index_offset_buf.Store4(sizeof(uint4) * thread_index, uint4(0, 0, 0, 0));
}