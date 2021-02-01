use super::buffer::{Buffer, BufferDesc};
use ash::{version::DeviceV1_0, vk};
use gpu_allocator::{AllocationCreateDesc, MemoryLocation, SubAllocation, VulkanAllocator};

use crate::{gpu_profiler::GpuProfilerQueryId, Device};

pub struct VkProfilerData {
    pub query_pool: vk::QueryPool,
    buffer: vk::Buffer,
    allocation: SubAllocation,
    next_query_id: std::sync::atomic::AtomicU32,
    gpu_profiler_query_ids: Vec<std::cell::Cell<GpuProfilerQueryId>>,
}

/*impl Drop for VkProfilerData {
    fn drop(&mut self) {
        unsafe {
            let vk = vk();
            vk.allocator
                .destroy_buffer(self.buffer, &self.allocation)
                .unwrap();

            vk.device.destroy_query_pool(self.query_pool, None);
        }

        let valid_query_count = self
            .next_query_id
            .load(std::sync::atomic::Ordering::Relaxed) as usize;

        crate::gpu_profiler::forget_queries(
            self.gpu_profiler_query_ids[0..valid_query_count]
                .iter()
                .map(std::cell::Cell::take),
        );
    }
}*/

const MAX_QUERY_COUNT: usize = 1024;

impl VkProfilerData {
    pub fn new(device: &ash::Device, allocator: &mut VulkanAllocator) -> Self {
        let usage: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_DST;

        /*let buffer = device
        .create_buffer(
            BufferDesc {
                size: MAX_QUERY_COUNT * 8 * 2,
                usage,
                mapped: true,
            },
            None,
        )
        .unwrap();*/

        let (buffer, allocation) = {
            let size = MAX_QUERY_COUNT * 8 * 2;
            let usage = vk::BufferUsageFlags::TRANSFER_DST;

            let buffer_info = vk::BufferCreateInfo {
                size: size as u64,
                usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let buffer = unsafe {
                device
                    .create_buffer(&buffer_info, None)
                    .expect("create_buffer")
            };
            let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

            let allocation = allocator
                .allocate(&AllocationCreateDesc {
                    name: "buffer",
                    requirements,
                    location: MemoryLocation::CpuToGpu,
                    linear: true, // Buffers are always linear
                })
                .unwrap();

            // Bind memory to the buffer
            unsafe {
                device
                    .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                    .expect("bind_buffer_memory")
            };

            (buffer, allocation)
        };

        let pool_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(MAX_QUERY_COUNT as u32 * 2);

        Self {
            query_pool: unsafe { device.create_query_pool(&pool_info, None) }
                .expect("create_query_pool"),
            buffer,
            allocation,
            next_query_id: Default::default(),
            gpu_profiler_query_ids: vec![
                std::cell::Cell::new(GpuProfilerQueryId::default());
                MAX_QUERY_COUNT
            ],
        }
    }

    pub fn get_query_id(&self, gpu_profiler_query_id: GpuProfilerQueryId) -> u32 {
        // TODO: handle running out of queries
        let id = self
            .next_query_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.gpu_profiler_query_ids[id as usize].set(gpu_profiler_query_id);
        id
    }

    // Two timing values per query
    pub fn retrieve_previous_result(&self) -> (Vec<GpuProfilerQueryId>, Vec<u64>) {
        let valid_query_count = self
            .next_query_id
            .load(std::sync::atomic::Ordering::Relaxed) as usize;

        let mapped_ptr = self.allocation.mapped_ptr().unwrap().as_ptr() as *const u64;
        let result =
            unsafe { std::slice::from_raw_parts(mapped_ptr, valid_query_count * 2) }.to_owned();

        (
            self.gpu_profiler_query_ids[0..valid_query_count]
                .iter()
                .map(std::cell::Cell::get)
                .collect(),
            result,
        )
    }

    pub fn begin_frame(&self, device: &Device, cmd: vk::CommandBuffer) {
        unsafe {
            device
                .raw
                .cmd_reset_query_pool(cmd, self.query_pool, 0, MAX_QUERY_COUNT as u32 * 2);
        }

        self.next_query_id
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn finish_frame(&self, device: &Device, cmd: vk::CommandBuffer) {
        let valid_query_count = self
            .next_query_id
            .load(std::sync::atomic::Ordering::Relaxed);

        unsafe {
            device.raw.cmd_copy_query_pool_results(
                cmd,
                self.query_pool,
                0,
                valid_query_count * 2,
                self.buffer,
                0,
                8,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
        }
    }
}
