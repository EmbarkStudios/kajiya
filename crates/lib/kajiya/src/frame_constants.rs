use crate::viewport::ViewConstants;

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
}
