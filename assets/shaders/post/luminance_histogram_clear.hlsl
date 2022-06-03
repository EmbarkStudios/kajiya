[[vk::binding(0)]] RWStructuredBuffer<uint> output_buffer;

[numthreads(256, 1, 1)]
void main(uint bin: SV_DispatchThreadID) {
    output_buffer[bin] = 0;
}
