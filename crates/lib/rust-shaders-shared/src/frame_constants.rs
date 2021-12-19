use crate::view_constants::ViewConstants;
use macaw::{IVec4, Vec4};

pub const MAX_CSGI_CASCADE_COUNT: usize = 4;

#[repr(C, align(16))]
#[derive(Copy, Clone, Default)]
pub struct GiCascadeConstants {
    pub scroll_frac: IVec4,
    pub scroll_int: IVec4,
    pub voxels_scrolled_this_frame: IVec4,
    pub volume_size: f32,
    pub voxel_size: f32,
    pub pad0: u32,
    pub pad1: u32,
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

    pub gi_cascades: [GiCascadeConstants; MAX_CSGI_CASCADE_COUNT],
}
