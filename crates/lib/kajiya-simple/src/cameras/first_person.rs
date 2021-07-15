use glam::{Mat4, Quat, Vec3};
use kajiya::camera::*;

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct FirstPersonCamera {
    // Degrees
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,

    pub position: Vec3,

    pub interp_rot: Quat,
    pub interp_pos: Vec3,

    pub move_smoothness: f32,
    pub look_smoothness: f32,
    pub move_speed: f32,

    interp_move_vec: Vec3,
}

pub struct FirstPersonCameraInput {
    pub move_vec: Vec3,
    pub yaw_delta: f32,
    pub pitch_delta: f32,
    pub dt: f32,
}

impl FirstPersonCamera {
    fn rotate_yaw(&mut self, delta: f32) {
        self.yaw = (self.yaw + delta) % 720_f32;
    }

    fn rotate_pitch(&mut self, delta: f32) {
        self.pitch += delta;
        if self.pitch < -90.0 {
            self.pitch = -90.0
        }
        if self.pitch > 90.0 {
            self.pitch = 90.0
        }
    }

    // https://stackoverflow.com/a/27497022
    fn three_axis_rot(r11: f32, r12: f32, r21: f32, r31: f32, r32: f32) -> [f32; 3] {
        [f32::atan2(r31, r32), f32::asin(r21), f32::atan2(r11, r12)]
    }

    // https://stackoverflow.com/a/27497022
    #[allow(dead_code)]
    pub fn look_at(&mut self, target: Vec3) {
        let q = Mat4::look_at_rh(self.interp_pos, target, Vec3::Y)
            .to_scale_rotation_translation()
            .1
            .conjugate()
            .normalize();

        let rots = Self::three_axis_rot(
            2.0 * (q.x * q.z + q.w * q.y),
            q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
            -2.0 * (q.y * q.z - q.w * q.x),
            2.0 * (q.x * q.y + q.w * q.z),
            q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
        );

        self.roll = rots[0].to_degrees();
        self.pitch = rots[1].to_degrees();
        self.yaw = rots[2].to_degrees();

        self.interp_rot = self.calc_rotation_quat();
    }

    fn translate(&mut self, local_v: Vec3) {
        //let rotation = self.calc_rotation_quat();
        let rotation = self.interp_rot;
        self.position += rotation * local_v * self.move_speed;
        //println!("self.position: {:?}", self.position);
    }

    fn calc_rotation_quat(&self) -> Quat {
        let yaw_rot: Quat = Quat::from_axis_angle(Vec3::Y, self.yaw.to_radians());
        let pitch_rot: Quat = Quat::from_axis_angle(Vec3::X, self.pitch.to_radians());
        let roll_rot: Quat = Quat::from_axis_angle(Vec3::Z, self.roll.to_radians());
        yaw_rot * (pitch_rot * roll_rot)
    }

    pub fn new(position: Vec3) -> FirstPersonCamera {
        FirstPersonCamera {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            position,
            interp_rot: Quat::from_axis_angle(Vec3::Y, -0.0f32.to_radians()),
            interp_pos: position,
            move_smoothness: 1.0,
            look_smoothness: 1.0,
            move_speed: 0.2,
            interp_move_vec: Vec3::ZERO,
        }
    }
}

impl FirstPersonCamera {
    pub fn update(&mut self, input: FirstPersonCameraInput) {
        let move_input_interp = 1.0 - (-input.dt * 30.0 / self.move_smoothness.max(1e-5)).exp();
        self.interp_move_vec = self.interp_move_vec.lerp(input.move_vec, move_input_interp);
        self.interp_move_vec *=
            Vec3::select(input.move_vec.cmpeq(Vec3::ZERO), Vec3::ZERO, Vec3::ONE);
        self.interp_move_vec *= Vec3::select(
            input.move_vec.signum().cmpeq(self.interp_move_vec.signum()),
            Vec3::ONE,
            -Vec3::ONE,
        );

        let move_dist = input.dt * 60.0;
        self.translate(self.interp_move_vec * move_dist);

        self.rotate_pitch(input.pitch_delta);
        self.rotate_yaw(input.yaw_delta);

        let target_quat = self.calc_rotation_quat();
        let rot_interp = 1.0 - (-input.dt * 30.0 / self.look_smoothness.max(1e-5)).exp();
        let pos_interp = 1.0 - (-input.dt * 16.0 / self.move_smoothness.max(1e-5)).exp();
        self.interp_rot = self.interp_rot.slerp(target_quat, rot_interp);
        self.interp_rot = self.interp_rot.normalize();
        self.interp_pos = self.interp_pos.lerp(self.position, pos_interp);
    }

    pub fn look(&self) -> CameraBodyMatrices {
        CameraBodyMatrices::from_position_rotation(self.interp_pos, self.interp_rot)
    }
}

/*#[derive(Clone)]
pub struct CameraConvergenceEnforcer<CameraType: Camera> {
    camera: CameraType,
    prev_matrices: CameraMatrices,
    frozen_matrices: CameraMatrices,
    prev_error: f32,
    is_converged: bool,
    pub convergence_sensitivity: f32,
}

#[allow(dead_code)]
impl<CameraType: Camera> CameraConvergenceEnforcer<CameraType> {
    pub fn new(camera: CameraType) -> Self {
        let matrices = camera.calc_matrices();
        Self {
            camera,
            prev_matrices: matrices,
            frozen_matrices: matrices,
            prev_error: -1.0,
            is_converged: true,
            convergence_sensitivity: 1.0,
        }
    }

    pub fn is_converged(&self) -> bool {
        self.is_converged
    }

    pub fn into_inner(self) -> CameraType {
        self.camera
    }
}

impl<CameraType: Camera> Camera for CameraConvergenceEnforcer<CameraType> {
    type InputType = CameraType::InputType;

    fn update<InputType: Into<Self::InputType>>(&mut self, input: InputType) {
        self.camera.update(input);

        let new_matrices = self.camera.calc_matrices();

        let cs_corners: [Vec4; 4] = [
            Vec4::new(-1.0, -1.0, 1e-5, 1.0),
            Vec4::new(1.0, -1.0, 1e-5, 1.0),
            Vec4::new(-1.0, 1.0, 1e-5, 1.0),
            Vec4::new(1.0, 1.0, 1e-5, 1.0),
        ];

        let clip_to_prev_clip = self.prev_matrices.view_to_clip
            * (self.prev_matrices.world_to_view * new_matrices.view_to_world)
            * new_matrices.clip_to_view;

        let error: f32 = cs_corners
            .iter()
            .copied()
            .map(|cs_cur| {
                let cs_prev = clip_to_prev_clip * cs_cur;
                let cs_prev = cs_prev * (1.0 / cs_prev.w);
                (cs_prev.truncate().truncate() - cs_cur.truncate().truncate()).length()
            })
            .sum();

        if error > 1e-5 * self.convergence_sensitivity
            || error
                > self.prev_error * (1.0 + 0.05 * self.convergence_sensitivity)
                    + 1e-5 * self.convergence_sensitivity
        {
            self.frozen_matrices = new_matrices;
            self.is_converged = false;
        } else {
            self.is_converged = true;
        }

        self.prev_matrices = new_matrices;
        self.prev_error = error;
    }

    fn calc_matrices(&self) -> CameraMatrices {
        self.frozen_matrices
    }
}
*/
