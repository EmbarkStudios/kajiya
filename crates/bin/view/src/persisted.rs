use kajiya_simple::{Quat, Vec3, Vec3Swizzles};

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SunState {
    // Not normalized. The scale is relevant to user input non-linearity.
    pub towards_sun: Vec3,
    pub size_multiplier: f32,
}

impl Default for SunState {
    fn default() -> Self {
        Self {
            towards_sun: Vec3::Y,
            size_multiplier: 1.0,
        }
    }
}

impl SunState {
    pub fn towards_sun(&self) -> Vec3 {
        self.towards_sun.normalize()
    }

    pub fn rotate(&mut self, ref_frame: &Quat, delta_x: f32, delta_y: f32) {
        // Project to the XZ plane, disregarding the Y component
        let mut xz = self.towards_sun.xz();

        // If the sun is below the horizon, we'll flip the movement direction
        let mut ysgn = if self.towards_sun.y >= 0.0 { 1.0 } else { -1.0 };

        const MOVE_SPEED: f32 = 0.2;

        // Working in projective geometry, add the new input
        xz += (*ref_frame * Vec3::new(-delta_x, 0.0, -delta_y) * (ysgn * MOVE_SPEED)).xz();

        // Our projective space is a unit disk with an associated sign. If we go outside the disk,
        // we wrap around and flip the sign, putting the sun on the other side of the horizon.
        {
            let len = xz.length();
            if len > 1.0 {
                ysgn *= -1.0;

                // Reflect off the edge of the disk
                xz *= (2.0 - len) / len;
            }
        }

        // Parabola-shaped Y projection giving flat and close-to-linear control
        // with the sun high above, and a gentle roll-off towards the horizon.
        const SQUISH: f32 = 0.3;
        let y = SQUISH * ysgn * (1.0 - xz.length_squared());

        self.towards_sun = Vec3::new(xz.x, y, xz.y);
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
            use_dynamic_adaptation: false,
        }
    }
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct PersistedState {
    pub camera: CameraState,
    pub exposure: ExposureState,
    pub light: LightState,
    pub movement: MovementState,
}
