pub mod backend;
pub mod bytes;
pub mod chunky_list;
pub mod dynamic_constants;
pub mod file;
pub mod pipeline_cache;
pub mod renderer;
pub mod rg;
pub mod shader_compiler;
pub mod transient_resource_cache;

pub use ash;
pub use backend::shader::{MAX_BINDLESS_DESCRIPTOR_COUNT, MAX_DESCRIPTOR_SETS};
pub use backend::{
    device::Device,
    image::{Image, ImageDesc, ImageSubResourceData},
    RenderBackend,
};
pub use vk_mem;
pub use vk_sync;

#[derive(Copy, Clone)]
pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
}

impl WindowConfig {
    pub fn dims(self) -> [u32; 2] {
        [self.width, self.height]
    }
}
