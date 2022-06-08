use crate::BackendError;

use super::device::Device;
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
    pub alignment: Option<u64>,
}

impl BufferDesc {
    pub fn new_gpu_only(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            memory_location: MemoryLocation::GpuOnly,
            alignment: None,
        }
    }

    pub fn new_cpu_to_gpu(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            memory_location: MemoryLocation::CpuToGpu,
            alignment: None,
        }
    }

    pub fn new_gpu_to_cpu(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            memory_location: MemoryLocation::GpuToCpu,
            alignment: None,
        }
    }

    pub fn alignment(mut self, alignment: u64) -> Self {
        self.alignment = Some(alignment);
        self
    }
}

impl Device {
    pub(crate) fn create_buffer_impl(
        raw: &ash::Device,
        allocator: &mut gpu_allocator::VulkanAllocator,
        desc: BufferDesc,
        name: &str,
    ) -> Result<Buffer, BackendError> {
        let buffer_info = vk::BufferCreateInfo {
            size: desc.size as u64,
            usage: desc.usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe {
            raw.create_buffer(&buffer_info, None)
                .expect("create_buffer")
        };
        let mut requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };

        if let Some(alignment) = desc.alignment {
            requirements.alignment = requirements.alignment.max(alignment);
        }

        // TODO: why does `get_buffer_memory_requirements` fail to get the correct alignment on AMD?
        if desc
            .usage
            .contains(vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR)
        {
            // TODO: query device props
            requirements.alignment = requirements.alignment.max(64);
        }

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: desc.memory_location,
                linear: true, // Buffers are always linear
            })
            .map_err(move |err| BackendError::Allocation {
                inner: err,
                name: name.to_owned(),
            })?;

        // Bind memory to the buffer
        unsafe {
            raw.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .expect("bind_buffer_memory")
        };

        Ok(Buffer {
            raw: buffer,
            desc,
            allocation,
        })
    }

    pub fn create_buffer(
        &self,
        mut desc: BufferDesc,
        name: impl Into<String>,
        initial_data: Option<&[u8]>,
    ) -> Result<Buffer, BackendError> {
        let name = name.into();

        if initial_data.is_some() {
            desc.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        let buffer =
            Self::create_buffer_impl(&self.raw, &mut self.global_allocator.lock(), desc, &name)?;

        if let Some(initial_data) = initial_data {
            let scratch_desc =
                BufferDesc::new_cpu_to_gpu(desc.size, vk::BufferUsageFlags::TRANSFER_SRC);

            let mut scratch_buffer = Self::create_buffer_impl(
                &self.raw,
                &mut self.global_allocator.lock(),
                scratch_desc,
                &format!("Initial data for {:?}", name),
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
            })?;
        }

        Ok(buffer)
    }

    pub fn immediate_destroy_buffer(&self, buffer: Buffer) {
        unsafe {
            self.raw.destroy_buffer(buffer.raw, None);
        }
        self.global_allocator
            .lock()
            .free(buffer.allocation)
            .expect("buffer memory deallocated");
    }
}
