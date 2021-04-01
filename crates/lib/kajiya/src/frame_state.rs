use glam::Vec3;

use crate::camera::CameraMatrices;

pub struct FrameState {
    pub camera_matrices: CameraMatrices,
    pub render_extent: [u32; 2],
    pub sun_direction: Vec3,
}
