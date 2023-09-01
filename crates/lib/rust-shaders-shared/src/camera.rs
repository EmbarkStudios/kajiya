use glam::{Mat4, Vec3, Vec4};

#[derive(PartialEq, Clone, Copy)]
#[repr(C)]
pub struct CameraMatrices {
    pub view_to_clip: Mat4,
    pub clip_to_view: Mat4,
    pub world_to_view: Mat4,
    pub view_to_world: Mat4,
}

impl CameraMatrices {
    pub fn eye_position(&self) -> Vec3 {
        (self.view_to_world * Vec4::new(0.0, 0.0, 0.0, 1.0)).truncate()
    }

    pub fn eye_direction(&self) -> Vec3 {
        (self.view_to_world * Vec4::new(0.0, 0.0, -1.0, 0.0))
            .truncate()
            .normalize()
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.view_to_clip.y_axis.y / self.view_to_clip.x_axis.x
    }
}
