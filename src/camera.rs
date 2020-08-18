use crate::math::*;

#[derive(PartialEq, Clone, Copy)]
pub struct CameraMatrices {
    pub view_to_clip: Mat4,
    pub clip_to_view: Mat4,
    pub world_to_view: Mat4,
    pub view_to_world: Mat4,
}

pub trait Camera {
    type InputType;

    fn update<InputType: Into<Self::InputType>>(&mut self, input: InputType);
    fn calc_matrices(&self) -> CameraMatrices;
}

impl<T: Camera> From<&T> for CameraMatrices {
    fn from(camera: &T) -> CameraMatrices {
        camera.calc_matrices()
    }
}

pub struct FirstPersonCamera {
    // Degrees
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub fov: f32,

    pub position: Vec3,
    pub near_dist: f32,
    pub aspect: f32,

    pub interp_rot: Quat,
    pub interp_pos: Vec3,

    pub move_smoothness: f32,
    pub look_smoothness: f32,
    pub move_speed: f32,
}

pub struct FirstPersonCameraInput {
    move_vec: Vec3,
    yaw_delta: f32,
    pitch_delta: f32,
    dt: f32,
}

/*impl<'a> From<&FrameState<'a>> for FirstPersonCameraInput {
    fn from(frame_state: &FrameState<'a>) -> FirstPersonCameraInput {
        let mut yaw_delta = 0.0;
        let mut pitch_delta = 0.0;

        if (frame_state.mouse.button_mask & 4) == 4 {
            yaw_delta = -0.1 * frame_state.mouse.delta.x();
            pitch_delta = -0.1 * frame_state.mouse.delta.y();
        }

        let mut move_vec = Vec3::zero();

        if frame_state.keys.is_down(VirtualKeyCode::W) {
            move_vec += Vec3::unit_z() * -1.0;
        }
        if frame_state.keys.is_down(VirtualKeyCode::S) {
            move_vec += Vec3::unit_z() * 1.0;
        }
        if frame_state.keys.is_down(VirtualKeyCode::A) {
            move_vec += Vec3::unit_x() * -1.0;
        }
        if frame_state.keys.is_down(VirtualKeyCode::D) {
            move_vec += Vec3::unit_x() * 1.0;
        }
        if frame_state.keys.is_down(VirtualKeyCode::Q) {
            move_vec += Vec3::unit_y() * -1.0;
        }
        if frame_state.keys.is_down(VirtualKeyCode::E) {
            move_vec += Vec3::unit_y() * 1.0;
        }

        if frame_state.keys.is_down(VirtualKeyCode::LControl) {
            move_vec *= 0.1;
        }

        FirstPersonCameraInput {
            move_vec,
            yaw_delta,
            pitch_delta,
            dt: frame_state.dt,
        }
    }
}*/

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

    fn translate(&mut self, local_v: Vec3) {
        let rotation = self.calc_rotation_quat();
        self.position += rotation * local_v * self.move_speed;
    }

    fn calc_rotation_quat(&self) -> Quat {
        let yaw_rot: Quat = Quat::from_axis_angle(Vec3::unit_y(), self.yaw.to_radians());
        let pitch_rot: Quat = Quat::from_axis_angle(Vec3::unit_x(), self.pitch.to_radians());
        let roll_rot: Quat = Quat::from_axis_angle(Vec3::unit_z(), self.roll.to_radians());
        yaw_rot * (pitch_rot * roll_rot)
    }

    pub fn new(position: Vec3) -> FirstPersonCamera {
        FirstPersonCamera {
            yaw: 0_f32,
            pitch: 0_f32,
            roll: 0_f32,
            fov: 45_f32,
            position,
            near_dist: 0.1_f32,
            aspect: 1.6_f32,
            interp_rot: Quat::from_axis_angle(Vec3::unit_y(), -0.0f32.to_radians()),
            interp_pos: position,
            move_smoothness: 1.0,
            look_smoothness: 1.0,
            move_speed: 12.0,
        }
    }
}

impl Camera for FirstPersonCamera {
    type InputType = FirstPersonCameraInput;

    fn update<InputType: Into<Self::InputType>>(&mut self, input: InputType) {
        let input = input.into();

        let move_dist = input.dt * 60.0;
        self.translate(input.move_vec * move_dist);

        self.rotate_pitch(input.pitch_delta);
        self.rotate_yaw(input.yaw_delta);

        let target_quat = self.calc_rotation_quat();
        let rot_interp = 1.0 - (-input.dt * 30.0 / self.look_smoothness.max(1e-5)).exp();
        let pos_interp = 1.0 - (-input.dt * 16.0 / self.move_smoothness.max(1e-5)).exp();
        self.interp_rot = self.interp_rot.slerp(target_quat, rot_interp);
        self.interp_rot.normalize();
        self.interp_pos = self.interp_pos.lerp(self.position, pos_interp).into();
    }

    fn calc_matrices(&self) -> CameraMatrices {
        let (view_to_clip, clip_to_view) = {
            let fov = self.fov.to_radians();
            let znear = self.near_dist;

            let h = (0.5 * fov).cos() / (0.5 * fov).sin();
            let w = h / self.aspect;

            (
                {
                    Mat4::from_cols(
                        Vec4::new(w, 0.0, 0.0, 0.0),
                        Vec4::new(0.0, h, 0.0, 0.0),
                        Vec4::new(0.0, 0.0, 0.0, -1.0),
                        Vec4::new(0.0, 0.0, znear, 0.0),
                    )

                    /*let mut m = Mat4::zero();
                    m.m11 = w;
                    m.m22 = h;
                    m.m34 = znear;
                    m.m43 = -1.0;
                    m*/
                },
                {
                    Mat4::from_cols(
                        Vec4::new(1.0 / w, 0.0, 0.0, 0.0),
                        Vec4::new(0.0, 1.0 / h, 0.0, 0.0),
                        Vec4::new(0.0, 0.0, 0.0, 1.0 / znear),
                        Vec4::new(0.0, 0.0, -1.0, 0.0),
                    )

                    /*let mut m = Mat4::zero();
                    m.m11 = 1.0 / w;
                    m.m22 = 1.0 / h;
                    m.m34 = -1.0;
                    m.m43 = 1.0 / znear;
                    m*/
                },
            )
        };

        let view_to_world = {
            let translation = Mat4::from_translation(self.interp_pos);
            translation * Mat4::from_quat(self.interp_rot)
        };

        let world_to_view = {
            let inv_translation = Mat4::from_translation(-self.interp_pos);
            Mat4::from_quat(self.interp_rot.conjugate()) * inv_translation
        };

        CameraMatrices {
            view_to_clip,
            clip_to_view,
            world_to_view,
            view_to_world,
        }
    }
}

pub struct CameraConvergenceEnforcer<CameraType: Camera> {
    camera: CameraType,
    prev_matrices: CameraMatrices,
    frozen_matrices: CameraMatrices,
    prev_error: f32,
    is_converged: bool,
}

#[allow(dead_code)]
impl<CameraType: Camera> CameraConvergenceEnforcer<CameraType> {
    pub fn new(camera: CameraType) -> Self {
        let matrices = camera.calc_matrices();
        Self {
            camera,
            prev_matrices: matrices.clone(),
            frozen_matrices: matrices,
            prev_error: -1.0,
            is_converged: true,
        }
    }

    pub fn is_converged(&self) -> bool {
        self.is_converged
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
                let cs_prev = cs_prev * (1.0 / cs_prev.w());
                (cs_prev.truncate().truncate() - cs_cur.truncate().truncate()).length()
            })
            .sum();

        if error > 1e-5 || error > self.prev_error * 1.05 + 1e-5 {
            self.frozen_matrices = new_matrices.clone();
            self.is_converged = false;
        } else {
            self.is_converged = true;
        }

        self.prev_matrices = new_matrices;
        self.prev_error = error;
    }

    fn calc_matrices(&self) -> CameraMatrices {
        self.frozen_matrices.clone()
    }
}
