use super::device::Device;
use anyhow::Result;
use ash::vk;
use gpu_allocator::{AllocationCreateDesc, MemoryLocation};

pub struct Buffer {
    pub raw: vk::Buffer,
    pub desc: BufferDesc,
    pub allocation: gpu_allocator::SubAllocation,
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
    pub memory_location: MemoryLocation,
}

impl BufferDesc {
    pub fn new_gpu_only(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            memory_location: MemoryLocation::GpuOnly,
        }
    }

    pub fn new_cpu_to_gpu(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            memory_location: MemoryLocation::CpuToGpu,
        }
    }

    pub fn new_gpu_to_cpu(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            memory_location: MemoryLocation::GpuToCpu,
        }
    }
}

impl Device {
    fn create_buffer_impl(&self, desc: BufferDesc) -> Result<Buffer> {
        let buffer_info = vk::BufferCreateInfo {
            size: desc.size as u64,
            usage: desc.usage,
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
                location: desc.memory_location,
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

    pub fn create_buffer(
        &self,
        mut desc: BufferDesc,
        initial_data: Option<&[u8]>,
    ) -> Result<Buffer> {
        if initial_data.is_some() {
            desc.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        let buffer = self.create_buffer_impl(desc)?;

        if let Some(initial_data) = initial_data {
            let scratch_desc = BufferDesc {
                size: desc.size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_location: MemoryLocation::CpuToGpu,
            };

            let mut scratch_buffer = self.create_buffer_impl(scratch_desc)?;

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
