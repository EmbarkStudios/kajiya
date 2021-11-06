struct VkDrawIndexedIndirectCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

[[vk::binding(0)]] RWStructuredBuffer<VkDrawIndexedIndirectCommand> bricks_meta;

[numthreads(1, 1, 1)]
void main() {
    //bricks_meta.Store(0, 0);
    VkDrawIndexedIndirectCommand cmd;
    cmd.index_count = 36;
    cmd.instance_count = 0;
    cmd.first_index = 0;
    cmd.vertex_offset = 0;
    cmd.first_instance = 0;
    bricks_meta[0] = cmd;
}
