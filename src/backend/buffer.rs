use super::device::Device;
use anyhow::Result;
use ash::vk;

pub struct Buffer {
    pub raw: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub allocation_info: vk_mem::AllocationInfo,
}

#[derive(Clone, Copy, Debug)]
pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
}

impl Device {
    pub fn create_buffer(&self, desc: BufferDesc) -> Result<Buffer> {
        let buffer_info = vk::BufferCreateInfo {
            // Allocate twice the size for even and odd frames
            size: desc.size as u64,
            usage: desc.usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer_mem_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (buffer, allocation, allocation_info) = self
            .global_allocator
            .create_buffer(&buffer_info, &buffer_mem_info)
            .expect("vma::create_buffer");

        Ok(Buffer {
            raw: buffer,
            allocation,
            allocation_info,
        })
    }
}
