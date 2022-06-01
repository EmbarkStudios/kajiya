use crate::{render_overrides::RenderOverrides, view_constants::ViewConstants};
use macaw::{IVec4, Vec4};

pub const IRCACHE_CASCADE_COUNT: usize = 12;

#[repr(C, align(16))]
#[derive(Copy, Clone, Default)]
pub struct IrcacheCascadeConstants {
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

    pub pre_exposure: f32,
    pub pre_exposure_prev: f32,
    pub pre_exposure_delta: f32,
    pub pad0: f32,

    pub render_overrides: RenderOverrides,

    pub ircache_grid_center: Vec4,
    pub ircache_cascades: [IrcacheCascadeConstants; IRCACHE_CASCADE_COUNT],
}
