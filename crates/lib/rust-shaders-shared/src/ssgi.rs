use macaw::Vec4;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SsgiConstants {
    pub input_tex_size: Vec4,
    pub output_tex_size: Vec4,
}

// Micro-occlusion settings used for denoising
pub const SSGI_HALF_SAMPLE_COUNT: u32 = 6;
pub const MAX_KERNEL_RADIUS_CS: f32 = 0.5;
pub const USE_KERNEL_DISTANCE_SCALING: bool = false;
pub const USE_RANDOM_JITTER: bool = false;

// Crazy settings for testing with cornell box
// pub const SSGI_HALF_SAMPLE_COUNT: u32 = 32;
// pub const MAX_KERNEL_RADIUS_CS: f32 = 100.0;
// pub const USE_KERNEL_DISTANCE_SCALING: bool = true;
// pub const USE_RANDOM_JITTER: bool = true;
