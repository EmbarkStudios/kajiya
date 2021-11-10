use crate::camera::CameraMatrices;
use macaw::{Mat4, UVec2, Vec2, Vec3};

#[derive(Clone, Copy)]
#[repr(C, align(16))]
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

impl ViewConstants {
    pub fn builder<CamMat: Into<CameraMatrices>>(
        camera_matrices: CamMat,
        prev_camera_matrices: CamMat,
        render_extent: [u32; 2],
    ) -> VieportConstantBuilder {
        VieportConstantBuilder {
            render_extent: render_extent.into(),
            camera_matrices: camera_matrices.into(),
            prev_camera_matrices: prev_camera_matrices.into(),
            pixel_offset: Vec2::ZERO,
        }
    }

    pub fn set_pixel_offset(&mut self, v: Vec2, render_extent: UVec2) {
        let sample_offset_pixels = v;
        let sample_offset_clip = Vec2::new(
            (2.0 * v.x) / render_extent.x as f32,
            (2.0 * v.y) / render_extent.y as f32,
        );

        let mut jitter_matrix = Mat4::IDENTITY;
        jitter_matrix.w_axis = (-sample_offset_clip).extend(0.0).extend(1.0);
        //jitter_matrix.m14 = -sample_offset_clip.x;
        //jitter_matrix.m24 = -sample_offset_clip.y;

        let mut jitter_matrix_inv = Mat4::IDENTITY;
        jitter_matrix_inv.w_axis = sample_offset_clip.extend(0.0).extend(1.0);
        //jitter_matrix_inv.m14 = sample_offset_clip.x;
        //jitter_matrix_inv.m24 = sample_offset_clip.y;

        let view_to_sample = jitter_matrix * self.view_to_clip;
        let sample_to_view = self.clip_to_view * jitter_matrix_inv;

        self.view_to_sample = view_to_sample;
        self.sample_to_view = sample_to_view;
        self.sample_offset_pixels = sample_offset_pixels;
        self.sample_offset_clip = sample_offset_clip;
    }

    pub fn eye_position(&self) -> Vec3 {
        let eye_pos_h = self.view_to_world.w_axis;
        eye_pos_h.truncate() / eye_pos_h.w
    }

    pub fn prev_eye_position(&self) -> Vec3 {
        let eye_pos_h = self.prev_view_to_prev_world.w_axis;
        eye_pos_h.truncate() / eye_pos_h.w
    }
}

pub struct VieportConstantBuilder {
    render_extent: UVec2,
    camera_matrices: CameraMatrices,
    prev_camera_matrices: CameraMatrices,
    pixel_offset: Vec2,
}

impl VieportConstantBuilder {
    #[allow(dead_code)]
    pub fn pixel_offset(mut self, v: Vec2) -> Self {
        self.pixel_offset = v;
        self
    }

    pub fn build(self) -> ViewConstants {
        let clip_to_prev_clip = self.prev_camera_matrices.view_to_clip
            * self.prev_camera_matrices.world_to_view
            * self.camera_matrices.view_to_world
            * self.camera_matrices.clip_to_view;

        let mut res = ViewConstants {
            view_to_clip: self.camera_matrices.view_to_clip,
            clip_to_view: self.camera_matrices.clip_to_view,
            view_to_sample: Mat4::ZERO,
            sample_to_view: Mat4::ZERO,
            world_to_view: self.camera_matrices.world_to_view,
            view_to_world: self.camera_matrices.view_to_world,

            clip_to_prev_clip,

            prev_view_to_prev_clip: self.prev_camera_matrices.view_to_clip,
            prev_clip_to_prev_view: self.prev_camera_matrices.clip_to_view,
            prev_world_to_prev_view: self.prev_camera_matrices.world_to_view,
            prev_view_to_prev_world: self.prev_camera_matrices.view_to_world,

            sample_offset_pixels: Vec2::ZERO,
            sample_offset_clip: Vec2::ZERO,
        };

        res.set_pixel_offset(self.pixel_offset, self.render_extent);
        res
    }
}
