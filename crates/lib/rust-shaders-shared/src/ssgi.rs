use glam::Vec4;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SsgiConstants {
    pub input_tex_size: Vec4,
    pub output_tex_size: Vec4,
    pub use_ao_only: u8,

    pub ssgi_half_sample_count: u32,
    pub max_kernel_radius_cs: f32,
    pub use_kernel_distance_scaling: u8,
    pub use_random_jitter: u8,
    pub kernel_radius: f32,
}

impl SsgiConstants {
    pub fn default_with_size(input_tex_size: Vec4, output_tex_size: Vec4) -> Self {
        Self {
            input_tex_size,
            output_tex_size,
            use_ao_only: 1,
            ssgi_half_sample_count: 6,
            max_kernel_radius_cs: 0.4,
            use_kernel_distance_scaling: 0,
            use_random_jitter: 0,
            kernel_radius: 60.0,
        }
    }

    pub fn insane_quality_with_size(input_tex_size: Vec4, output_tex_size: Vec4) -> Self {
        Self {
            input_tex_size,
            output_tex_size,
            use_ao_only: 0,
            ssgi_half_sample_count: 32,
            max_kernel_radius_cs: 100.0,
            use_kernel_distance_scaling: 1,
            use_random_jitter: 1,
            kernel_radius: 5.0,
        }
    }
}
