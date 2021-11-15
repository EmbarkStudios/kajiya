use glam::Vec3;

use rust_shaders_shared::camera::CameraMatrices;

pub struct WorldFrameDesc {
    pub camera_matrices: CameraMatrices,
    pub render_extent: [u32; 2],
    pub sun_direction: Vec3,
}
