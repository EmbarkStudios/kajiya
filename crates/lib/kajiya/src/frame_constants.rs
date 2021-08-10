use crate::viewport::ViewConstants;

pub const MAX_CSGI_CASCADE_COUNT: usize = 4;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct GiCascadeConstants {
    pub scroll_frac: [i32; 4],
    pub scroll_int: [i32; 4],
    pub voxels_scrolled_this_frame: [i32; 4],
    pub volume_size: f32,
    pub voxel_size: f32,
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FrameConstants {
    pub view_constants: ViewConstants,

    pub sun_direction: [f32; 4],

    pub frame_idx: u32,
    pub delta_time_seconds: f32,
    pub sun_angular_radius_cos: f32,
    pub global_fog_thickness: f32,

    pub sun_color_multiplier: [f32; 4],
    pub sky_ambient: [f32; 4],

    pub triangle_light_count: u32,
    pub world_gi_scale: f32,
    pub pad: [u32; 2],

    pub gi_cascades: [GiCascadeConstants; MAX_CSGI_CASCADE_COUNT],
}
