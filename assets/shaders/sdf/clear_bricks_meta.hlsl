[[vk::binding(0)]] RWByteAddressBuffer bricks_meta;

[numthreads(1, 1, 1)]
void main(in uint3 pix: SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    bricks_meta.Store(0, 0);
}
