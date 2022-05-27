use kajiya_simple::{Quat, Vec3};

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SunState {
    pub theta: f32,
    pub phi: f32,
}

impl SunState {
    pub fn direction(&self) -> Vec3 {
        fn spherical_to_cartesian(theta: f32, phi: f32) -> Vec3 {
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            Vec3::new(x, y, z)
        }

        spherical_to_cartesian(self.theta, self.phi)
    }
}

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LocalLightsState {
    pub theta: f32,
    pub phi: f32,
    pub count: u32,
    pub distance: f32,
    pub multiplier: f32,
}

trait ShouldResetPathTracer {
    fn should_reset_path_tracer(&self, other: &Self) -> bool;
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CameraState {
    pub position: Vec3,
    pub rotation: Quat,
    pub vertical_fov: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: Vec3::ONE,
            rotation: Quat::IDENTITY,
            vertical_fov: 62.0,
        }
    }
}

impl ShouldResetPathTracer for CameraState {
    fn should_reset_path_tracer(&self, other: &Self) -> bool {
        !self.position.abs_diff_eq(other.position, 1e-5)
            || !self.rotation.abs_diff_eq(other.rotation, 1e-5)
            || self.vertical_fov != other.vertical_fov
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LightState {
    pub emissive_multiplier: f32,
    pub enable_emissive: bool,
    pub sun: SunState,
    pub local_lights: LocalLightsState,
}

impl Default for LightState {
    fn default() -> Self {
        Self {
            emissive_multiplier: 1.0,
            enable_emissive: true,
            sun: SunState {
                theta: -4.54,
                phi: 1.48,
            },
            local_lights: LocalLightsState {
                theta: 1.0,
                phi: 1.0,
                count: 0,
                distance: 1.5,
                multiplier: 10.0,
            },
        }
    }
}

impl ShouldResetPathTracer for LightState {
    fn should_reset_path_tracer(&self, other: &Self) -> bool {
        self.emissive_multiplier != other.emissive_multiplier
            || self.sun != other.sun
            || self.local_lights != other.local_lights
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MovementState {
    pub camera_speed: f32,
    pub camera_smoothness: f32,
    pub sun_rotation_smoothness: f32,
}

impl Default for MovementState {
    fn default() -> Self {
        Self {
            camera_speed: 2.5,
            camera_smoothness: 1.0,
            sun_rotation_smoothness: 1.0,
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ExposureState {
    pub ev_shift: f32,
    #[serde(default)]
    pub use_dynamic_adaptation: bool,
}

impl Default for ExposureState {
    fn default() -> Self {
        Self {
            ev_shift: 0.0,
            use_dynamic_adaptation: true,
        }
    }
}
