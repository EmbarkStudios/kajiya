struct VkDrawIndirectCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

[[vk::binding(0)]] RWStructuredBuffer<VkDrawIndirectCommand> bricks_meta;

[numthreads(1, 1, 1)]
void main(in uint3 pix: SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    //bricks_meta.Store(0, 0);
    VkDrawIndirectCommand cmd;
    cmd.vertexCount = 3;
    cmd.instanceCount = 0;
    cmd.firstVertex = 0;
    cmd.firstInstance = 0;
    bricks_meta[0] = cmd;
}
