use super::device::Device;
use anyhow::Result;
use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc},
    MemoryLocation,
};

pub struct Buffer {
    pub raw: vk::Buffer,
    pub desc: BufferDesc,
    pub allocation: Allocation,
}

impl Buffer {
    pub fn device_address(&self, device: &Device) -> u64 {
        unsafe {
            device.raw.get_buffer_device_address(
                &ash::vk::BufferDeviceAddressInfo::builder().buffer(self.raw),
            )
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
    pub mapped: bool,
}

impl BufferDesc {
    pub fn new(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            mapped: false,
        }
    }
}

impl Device {
    // TODO: not pub.
    pub fn create_buffer_impl(
        &self,
        desc: BufferDesc,
        extra_usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
    ) -> Result<Buffer> {
        let buffer_info = vk::BufferCreateInfo {
            size: desc.size as u64,
            usage: desc.usage | extra_usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe {
            self.raw
                .create_buffer(&buffer_info, None)
                .expect("create_buffer")
        };
        let mut requirements = unsafe { self.raw.get_buffer_memory_requirements(buffer) };

        // TODO: why does `get_buffer_memory_requirements` fail to get the correct alignment on AMD?
        if desc
            .usage
            .contains(vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR)
        {
            // TODO: query device props
            requirements.alignment = requirements.alignment.max(64);
        }

        let allocation = self
            .global_allocator
            .lock()
            .allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: memory_location,
                linear: true, // Buffers are always linear
            })?;

        // Bind memory to the buffer
        unsafe {
            self.raw
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .expect("bind_buffer_memory")
        };

        /*let (buffer, allocation, allocation_info) = self
        .global_allocator
        .create_buffer(&buffer_info, &buffer_mem_info)
        .expect("vma::create_buffer");*/

        Ok(Buffer {
            raw: buffer,
            desc,
            allocation,
        })
    }

    pub fn create_buffer(&self, desc: BufferDesc, initial_data: Option<&[u8]>) -> Result<Buffer> {
        let memory_location = if desc.mapped {
            MemoryLocation::CpuToGpu
        } else {
            MemoryLocation::GpuOnly
        };

        let buffer = self.create_buffer_impl(
            desc,
            if initial_data.is_some() {
                vk::BufferUsageFlags::TRANSFER_DST
            } else {
                vk::BufferUsageFlags::empty()
            },
            memory_location,
        )?;

        if let Some(initial_data) = initial_data {
            let mut scratch_buffer = self.create_buffer_impl(
                desc,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            )?;

            scratch_buffer.allocation.mapped_slice_mut().unwrap()[0..initial_data.len()]
                .copy_from_slice(initial_data);

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
