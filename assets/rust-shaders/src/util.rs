use crate::frame_constants::FrameConstants;
use spirv_std::glam::{UVec2, Vec2, Vec4, Vec4Swizzles};

pub fn get_uv_u(pix: UVec2, tex_size: Vec4) -> Vec2 {
    (pix.as_f32() + Vec2::splat(0.5)) * tex_size.zw()
}

pub fn depth_to_view_z(depth: f32, frame_constants: &FrameConstants) -> f32 {
    (depth
        * -frame_constants
            .view_constants
            .clip_to_view
            .to_cols_array_2d()[2][3])
        .recip()
}
