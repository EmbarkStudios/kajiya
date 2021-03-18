pub mod bytes;
pub mod chunky_list;
pub mod dynamic_constants;
pub mod file;
pub mod gpu_profiler;
pub mod pipeline_cache;
pub mod shader_compiler;
pub mod transient_resource_cache;
pub mod vulkan;

pub use ash;
pub use gpu_allocator;
pub use rspirv_reflect;
pub use vk_sync;
pub use vulkan::shader::{MAX_BINDLESS_DESCRIPTOR_COUNT, MAX_DESCRIPTOR_SETS};
pub use vulkan::{device::Device, image::*, RenderBackend};

#[derive(Copy, Clone)]
pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
    pub vsync: bool,
}

impl WindowConfig {
    pub fn dims(self) -> [u32; 2] {
        [self.width, self.height]
    }
}
