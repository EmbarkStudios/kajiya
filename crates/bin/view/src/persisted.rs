use kajiya_simple::{Mat2, Quat, Vec2, Vec3, Vec3Swizzles};

use crate::misc::smoothstep;

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SunState {
    pub controller_xz: Vec2,
    pub size_multiplier: f32,
}

impl Default for SunState {
    fn default() -> Self {
        Self {
            controller_xz: Vec2::ZERO,
            size_multiplier: 1.0,
        }
    }
}

impl SunState {
    pub fn towards_sun(&self) -> Vec3 {
        let mut xz = self.controller_xz;
        let len = xz.length();

        let ysgn = if len > 1.0 {
            // First outer ring: the sun goes below the horizon
            xz *= (2.0 - len) / len;
            -1.0
        } else {
            // Inner disk
            1.0
        };

        const SQUISH: f32 = 0.2;
        let y = SQUISH * ysgn * (1.0 - xz.length_squared());
        Vec3::new(xz.x, y, xz.y).normalize()
    }

    pub fn rotate(&mut self, ref_frame: &Quat, delta_x: f32, delta_y: f32) {
        let mut xz = self.controller_xz;

        const MOVE_SPEED: f32 = 0.2;

        let xz_norm = xz.normalize_or_zero();

        // The controller has a singularity in the second outer ring.
        // This rotation will kick it out.
        let rotation_strength = smoothstep(1.2, 1.5, xz.length());
        let delta = *ref_frame * Vec3::new(-delta_x, 0.0, -delta_y);
        let move_align = delta.xz().perp_dot(xz_norm);

        // Working in projective geometry, add the new input
        xz += (delta * MOVE_SPEED).xz();

        let rm = Mat2::from_angle(move_align * rotation_strength);
        xz = rm * xz;

        {
            let len = xz.length();
            if len > 2.0 {
                // Second outer ring; reflect to the other side.
                xz *= -(4.0 - len) / len;
            }
        }

        self.controller_xz = xz;
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

pub trait ShouldResetPathTracer {
    fn should_reset_path_tracer(&self, _: &Self) -> bool {
        false
    }
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
            sun: SunState::default(),
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
            || self.enable_emissive != other.enable_emissive
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
            sun_rotation_smoothness: 0.0,
        }
    }
}

impl ShouldResetPathTracer for MovementState {}

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
            use_dynamic_adaptation: false,
        }
    }
}

impl ShouldResetPathTracer for ExposureState {}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PersistedState {
    pub camera: CameraState,
    pub light: LightState,
    pub exposure: ExposureState,
    pub movement: MovementState,
}

impl ShouldResetPathTracer for PersistedState {
    fn should_reset_path_tracer(&self, other: &Self) -> bool {
        self.camera.should_reset_path_tracer(&other.camera)
            || self.exposure.should_reset_path_tracer(&other.exposure)
            || self.light.should_reset_path_tracer(&other.light)
            || self.movement.should_reset_path_tracer(&other.movement)
    }
}
