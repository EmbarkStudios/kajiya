use ash::vk;
pub struct Buffer {
    pub raw: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub allocation_info: vk_mem::AllocationInfo,
}
