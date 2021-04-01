use glam::Vec3;
use kajiya_backend::WindowConfig;

use crate::camera::CameraMatrices;

pub struct FrameState {
    pub camera_matrices: CameraMatrices,
    pub window_cfg: WindowConfig,
    //pub swapchain_extent: [u32; 2],
    pub sun_direction: Vec3,
}
