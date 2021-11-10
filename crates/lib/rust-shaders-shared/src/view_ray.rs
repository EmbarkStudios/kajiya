use crate::frame_constants::FrameConstants;
use crate::util::*;
use macaw::*;

pub struct ViewRayContext {
    pub ray_dir_cs: Vec4,
    pub ray_dir_vs_h: Vec4,
    pub ray_dir_ws_h: Vec4,

    pub ray_origin_cs: Vec4,
    pub ray_origin_vs_h: Vec4,
    pub ray_origin_ws_h: Vec4,

    pub ray_hit_cs: Vec4,
    pub ray_hit_vs_h: Vec4,
    pub ray_hit_ws_h: Vec4,
}

impl ViewRayContext {
    pub fn ray_dir_vs(&self) -> Vec3 {
        self.ray_dir_vs_h.xyz()
    }

    pub fn ray_dir_ws(&self) -> Vec3 {
        self.ray_dir_ws_h.xyz()
    }

    pub fn ray_origin_vs(&self) -> Vec3 {
        self.ray_origin_vs_h.xyz() / self.ray_origin_vs_h.w
    }

    pub fn ray_origin_ws(&self) -> Vec3 {
        self.ray_origin_ws_h.xyz() / self.ray_origin_ws_h.w
    }

    pub fn ray_hit_vs(&self) -> Vec3 {
        self.ray_hit_vs_h.xyz() / self.ray_hit_vs_h.w
    }

    pub fn ray_hit_ws(&self) -> Vec3 {
        self.ray_hit_ws_h.xyz() / self.ray_hit_ws_h.w
    }

    pub fn from_uv(uv: Vec2, frame_constants: &FrameConstants) -> Self {
        let view_constants = frame_constants.view_constants;

        let ray_dir_cs = uv_to_cs(uv).extend(0.0).extend(1.0);
        let ray_dir_vs_h = view_constants.sample_to_view * ray_dir_cs;
        let ray_dir_ws_h = view_constants.view_to_world * ray_dir_vs_h;

        let ray_origin_cs = uv_to_cs(uv).extend(1.0).extend(1.0);
        let ray_origin_vs_h = view_constants.sample_to_view * ray_origin_cs;
        let ray_origin_ws_h = view_constants.view_to_world * ray_origin_vs_h;

        ViewRayContext {
            ray_dir_cs,
            ray_dir_vs_h,
            ray_dir_ws_h,
            ray_origin_cs,
            ray_origin_vs_h,
            ray_origin_ws_h,
            ray_hit_cs: Vec4::ZERO,
            ray_hit_vs_h: Vec4::ZERO,
            ray_hit_ws_h: Vec4::ZERO,
        }
    }

    pub fn from_uv_and_depth(uv: Vec2, depth: f32, frame_constants: &FrameConstants) -> Self {
        let view_constants = frame_constants.view_constants;

        let ray_dir_cs = uv_to_cs(uv).extend(0.0).extend(1.0);
        let ray_dir_vs_h = view_constants.sample_to_view * ray_dir_cs;
        let ray_dir_ws_h = view_constants.view_to_world * ray_dir_vs_h;

        let ray_origin_cs = uv_to_cs(uv).extend(1.0).extend(1.0);
        let ray_origin_vs_h = view_constants.sample_to_view * ray_origin_cs;
        let ray_origin_ws_h = view_constants.view_to_world * ray_origin_vs_h;

        let ray_hit_cs = uv_to_cs(uv).extend(depth).extend(1.0);
        let ray_hit_vs_h = view_constants.sample_to_view * ray_hit_cs;
        let ray_hit_ws_h = view_constants.view_to_world * ray_hit_vs_h;

        ViewRayContext {
            ray_dir_cs,
            ray_dir_vs_h,
            ray_dir_ws_h,
            ray_origin_cs,
            ray_origin_vs_h,
            ray_origin_ws_h,
            ray_hit_cs,
            ray_hit_vs_h,
            ray_hit_ws_h,
        }
    }
}
