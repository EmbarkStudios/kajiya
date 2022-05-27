use glam::Vec3;

use rust_shaders_shared::camera::CameraMatrices;

pub struct WorldFrameDesc {
    pub camera_matrices: CameraMatrices,

    /// Internal render resolution, before any upsampling
    pub render_extent: [u32; 2],

    /// Direction _towards_ the sun.
    pub sun_direction: Vec3,
}
