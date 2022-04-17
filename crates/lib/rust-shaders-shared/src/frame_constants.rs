use crate::view_constants::ViewConstants;
use macaw::{IVec4, Vec4};

pub const RCACHE_CASCADE_COUNT: usize = 12;

#[repr(C, align(16))]
#[derive(Copy, Clone, Default)]
pub struct RcacheCascadeConstants {
    pub origin: IVec4,
    pub voxels_scrolled_this_frame: IVec4,
}

#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct FrameConstants {
    pub view_constants: ViewConstants,

    pub sun_direction: Vec4,

    pub frame_index: u32,
    pub delta_time_seconds: f32,
    pub sun_angular_radius_cos: f32,
    pub triangle_light_count: u32,

    pub sun_color_multiplier: Vec4,
    pub sky_ambient: Vec4,

    pub world_gi_scale: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,

    pub rcache_cascades: [RcacheCascadeConstants; RCACHE_CASCADE_COUNT],
}
