use crate::{
    camera::CameraMatrices,
    math::{Mat4, Vec2},
};

#[derive(Clone, Copy)]
#[repr(C)]
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
        width: u32,
        height: u32,
    ) -> VieportConstantBuilder {
        VieportConstantBuilder {
            width,
            height,
            camera_matrices: camera_matrices.into(),
            prev_camera_matrices: prev_camera_matrices.into(),
            pixel_offset: Vec2::zero(),
        }
    }

    pub fn set_pixel_offset(&mut self, v: Vec2, width: u32, height: u32) {
        let sample_offset_pixels = v;
        let sample_offset_clip =
            Vec2::new((2.0 * v.x()) / width as f32, (2.0 * v.y()) / height as f32);

        let mut jitter_matrix = Mat4::identity();
        jitter_matrix.set_w_axis((-sample_offset_clip).extend(0.0).extend(1.0));
        //jitter_matrix.m14 = -sample_offset_clip.x;
        //jitter_matrix.m24 = -sample_offset_clip.y;

        let mut jitter_matrix_inv = Mat4::identity();
        jitter_matrix_inv.set_w_axis(sample_offset_clip.extend(0.0).extend(1.0));
        //jitter_matrix_inv.m14 = sample_offset_clip.x;
        //jitter_matrix_inv.m24 = sample_offset_clip.y;

        let view_to_sample = jitter_matrix * self.view_to_clip;
        let sample_to_view = self.clip_to_view * jitter_matrix_inv;

        self.view_to_sample = view_to_sample;
        self.sample_to_view = sample_to_view;
        self.sample_offset_pixels = sample_offset_pixels;
        self.sample_offset_clip = sample_offset_clip;
    }
}

pub struct VieportConstantBuilder {
    width: u32,
    height: u32,
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
            view_to_sample: Mat4::zero(),
            sample_to_view: Mat4::zero(),
            world_to_view: self.camera_matrices.world_to_view,
            view_to_world: self.camera_matrices.view_to_world,

            clip_to_prev_clip,

            prev_view_to_prev_clip: self.prev_camera_matrices.view_to_clip,
            prev_clip_to_prev_view: self.prev_camera_matrices.clip_to_view,
            prev_world_to_prev_view: self.prev_camera_matrices.world_to_view,
            prev_view_to_prev_world: self.prev_camera_matrices.view_to_world,

            sample_offset_pixels: Vec2::zero(),
            sample_offset_clip: Vec2::zero(),
        };

        res.set_pixel_offset(self.pixel_offset, self.width, self.height);
        res
    }
}
