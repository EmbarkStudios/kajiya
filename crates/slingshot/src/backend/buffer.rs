use super::device::Device;
use anyhow::Result;
use ash::{version::DeviceV1_0, vk};

pub struct Buffer {
    pub raw: vk::Buffer,
    pub desc: BufferDesc,
    pub allocation: vk_mem::Allocation,
    pub allocation_info: vk_mem::AllocationInfo,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
    pub mapped: bool,
}

impl Device {
    pub(crate) fn create_buffer_impl(
        &self,
        desc: BufferDesc,
        extra_usage: vk::BufferUsageFlags,
        buffer_mem_info: vk_mem::AllocationCreateInfo,
    ) -> Result<Buffer> {
        let buffer_info = vk::BufferCreateInfo {
            size: desc.size as u64,
            usage: desc.usage | extra_usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let (buffer, allocation, allocation_info) = self
            .global_allocator
            .create_buffer(&buffer_info, &buffer_mem_info)
            .expect("vma::create_buffer");

        Ok(Buffer {
            raw: buffer,
            desc,
            allocation,
            allocation_info,
        })
    }

    pub fn create_buffer(&self, desc: BufferDesc, initial_data: Option<&[u8]>) -> Result<Buffer> {
        let (memory_usage, allocation_create_flags) = if desc.mapped {
            (
                vk_mem::MemoryUsage::CpuToGpu,
                vk_mem::AllocationCreateFlags::MAPPED,
            )
        } else {
            (
                vk_mem::MemoryUsage::GpuOnly,
                vk_mem::AllocationCreateFlags::NONE,
            )
        };

        let buffer = self.create_buffer_impl(
            desc,
            if initial_data.is_some() {
                vk::BufferUsageFlags::TRANSFER_DST
            } else {
                vk::BufferUsageFlags::empty()
            },
            vk_mem::AllocationCreateInfo {
                usage: memory_usage,
                flags: allocation_create_flags,
                ..Default::default()
            },
        )?;

        if let Some(initial_data) = initial_data {
            let scratch_buffer = self.create_buffer_impl(
                desc,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::CpuToGpu,
                    flags: vk_mem::AllocationCreateFlags::MAPPED,
                    ..Default::default()
                },
            )?;
            unsafe {
                std::slice::from_raw_parts_mut(
                    scratch_buffer.allocation_info.get_mapped_data(),
                    desc.size,
                )
            }
            .copy_from_slice(&initial_data);

            self.with_setup_cb(|cb| unsafe {
                self.raw.cmd_copy_buffer(
                    cb,
                    scratch_buffer.raw,
                    buffer.raw,
                    &[ash::vk::BufferCopy::builder()
                        .dst_offset(0)
                        .src_offset(0)
                        .size(desc.size as u64)
                        .build()],
                );
            });
        }

        Ok(buffer)
    }
}
