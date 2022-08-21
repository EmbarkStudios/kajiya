use std::path::PathBuf;

use kajiya::world_renderer::InstanceHandle;
use kajiya_simple::{Affine3A, EulerRot, Mat2, Quat, Vec2, Vec3, Vec3Swizzles};

use crate::{misc::smoothstep, sequence::Sequence};

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SunState {
    pub controller: SunController,
    pub size_multiplier: f32,
}

impl Default for SunState {
    fn default() -> Self {
        Self {
            controller: SunController::default(),
            size_multiplier: 1.0,
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SunController {
    #[serde(skip)]
    latent: Option<Vec2>,
    towards_sun: Vec3,
}

impl PartialEq for SunController {
    fn eq(&self, other: &Self) -> bool {
        self.towards_sun == other.towards_sun
    }
}

impl Default for SunController {
    fn default() -> Self {
        Self {
            latent: None,
            towards_sun: Vec3::Y,
        }
    }
}

const SUN_CONTROLLER_SQUISH: f32 = 0.2;

impl SunController {
    pub fn towards_sun(&self) -> Vec3 {
        self.towards_sun
    }

    #[allow(dead_code)]
    pub fn set_towards_sun(&mut self, towards_sun: Vec3) {
        self.towards_sun = towards_sun;
        self.latent = None;
    }

    fn calculate_towards_sun(latent: Vec2) -> Vec3 {
        let mut xz = latent;
        let len = xz.length();

        let ysgn = if len > 1.0 {
            xz *= (2.0 - len) / len;
            -1.0
        } else {
            1.0
        };

        let y = SUN_CONTROLLER_SQUISH * ysgn * (1.0 - xz.length_squared());
        Vec3::new(xz.x, y, xz.y).normalize()
    }

    fn calculate_latent(towards_sun: Vec3) -> Vec2 {
        let t = SUN_CONTROLLER_SQUISH;
        let t2 = t * t;

        let y2 = towards_sun.y * towards_sun.y;
        let y4 = y2 * y2;

        // Mathematica goes brrrrrrr
        let a = -y2 + 2.0 * t2 * (-1.0 + y2) + (y4 - 4.0 * t2 * y2 * (-1.0 + y2)).sqrt();
        let b = 2.0 * t2 * (-1.0 + y2);
        let xz_len = (a / b).sqrt();

        let xz_len = if xz_len.is_finite() { xz_len } else { 0.0 };

        let mut xz = towards_sun.xz() * (xz_len / towards_sun.xz().length().max(1e-10));

        if towards_sun.y < 0.0 {
            xz *= (2.0 - xz_len) / xz_len;
        }

        xz
    }

    pub fn view_space_rotate(&mut self, ref_frame: &Quat, delta_x: f32, delta_y: f32) {
        let mut xz = *self
            .latent
            .get_or_insert_with(|| Self::calculate_latent(self.towards_sun));

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

        self.latent = Some(xz);
        self.towards_sun = Self::calculate_towards_sun(xz);
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

fn default_contrast() -> f32 {
    1.0
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ExposureState {
    pub ev_shift: f32,
    #[serde(default)]
    pub use_dynamic_adaptation: bool,
    #[serde(default)]
    pub dynamic_adaptation_speed: f32,
    #[serde(default)]
    pub dynamic_adaptation_low_clip: f32,
    #[serde(default)]
    pub dynamic_adaptation_high_clip: f32,
    #[serde(default = "default_contrast")]
    pub contrast: f32,
}

impl Default for ExposureState {
    fn default() -> Self {
        Self {
            ev_shift: 0.0,
            use_dynamic_adaptation: false,
            dynamic_adaptation_speed: 0.0,
            dynamic_adaptation_low_clip: 0.0,
            dynamic_adaptation_high_clip: 0.0,
            contrast: default_contrast(),
        }
    }
}

impl ShouldResetPathTracer for ExposureState {}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SceneElementTransform {
    pub position: Vec3,
    pub rotation_euler_degrees: Vec3,
    pub scale: Vec3,
}

impl SceneElementTransform {
    pub const IDENTITY: SceneElementTransform = SceneElementTransform {
        position: Vec3::ZERO,
        rotation_euler_degrees: Vec3::ZERO,
        scale: Vec3::ONE,
    };

    pub fn affine_transform(&self) -> Affine3A {
        Affine3A::from_scale_rotation_translation(
            self.scale,
            Quat::from_euler(
                EulerRot::YXZ,
                self.rotation_euler_degrees.y.to_radians(),
                self.rotation_euler_degrees.x.to_radians(),
                self.rotation_euler_degrees.z.to_radians(),
            ),
            self.position,
        )
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum MeshSource {
    File(PathBuf),
    Cache(PathBuf),
}

#[derive(Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SceneElement {
    #[serde(skip)]
    pub instance: InstanceHandle,

    pub source: MeshSource,
    pub transform: SceneElementTransform,
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SceneState {
    pub elements: Vec<SceneElement>,

    #[serde(default)]
    pub ibl: Option<PathBuf>,
}

impl ShouldResetPathTracer for SceneState {
    fn should_reset_path_tracer(&self, other: &Self) -> bool {
        self.elements != other.elements
    }
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PersistedState {
    pub camera: CameraState,
    pub light: LightState,
    pub exposure: ExposureState,
    pub movement: MovementState,
    pub sequence: Sequence,
    #[serde(default)]
    pub scene: SceneState,
}

impl ShouldResetPathTracer for PersistedState {
    fn should_reset_path_tracer(&self, other: &Self) -> bool {
        self.camera.should_reset_path_tracer(&other.camera)
            || self.exposure.should_reset_path_tracer(&other.exposure)
            || self.light.should_reset_path_tracer(&other.light)
            || self.movement.should_reset_path_tracer(&other.movement)
            || self.scene.should_reset_path_tracer(&other.scene)
    }
}
