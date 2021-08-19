use spirv_std::glam::{Mat4, UVec4, Vec2, Vec4};

pub const MAX_CSGI_CASCADE_COUNT: usize = 4;

#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct ViewConstants {
    pub view_to_clip: Mat4,
    pub clip_to_view: Mat4,
    pub view_to_sample: Mat4,
    pub sample_to_view: Mat4,
    pub world_to_view: Mat4,
    pub view_to_world: Mat4,

    pub clip_to_prev_clip: Mat4,

    pub prev_view_to_prev_clip: Mat4,
    pub prev_clip_to_prev_view: Mat4,
    pub prev_world_to_prev_view: Mat4,
    pub prev_view_to_prev_world: Mat4,

    pub sample_offset_pixels: Vec2,
    pub sample_offset_clip: Vec2,
}

#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct GiCascadeConstants {
    pub scroll_frac: UVec4,
    pub scroll_int: UVec4,
    pub voxels_scrolled_this_frame: UVec4,
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

    pub frame_idx: u32,
    pub delta_time_seconds: f32,
    pub sun_angular_radius_cos: f32,
    pub global_fog_thickness: f32,

    pub sun_color_multiplier: Vec4,
    pub sky_ambient: Vec4,

    pub triangle_light_count: u32,
    pub world_gi_scale: f32,
    pub pad0: u32,
    pub pad1: u32,

    pub gi_cascades: [GiCascadeConstants; MAX_CSGI_CASCADE_COUNT],
}
