use glam::Vec3;
use kajiya::camera::*;
use winit::event::VirtualKeyCode;

use crate::input::InputState;

impl From<&InputState> for FirstPersonCameraInput {
    fn from(input_state: &InputState) -> FirstPersonCameraInput {
        let mut yaw_delta = 0.0;
        let mut pitch_delta = 0.0;

        if (input_state.mouse.button_mask & 4) == 4 {
            yaw_delta = -0.1 * input_state.mouse.delta.x;
            pitch_delta = -0.1 * input_state.mouse.delta.y;
        }

        let mut move_vec = Vec3::zero();

        if input_state.keys.is_down(VirtualKeyCode::W) {
            move_vec += Vec3::unit_z() * -1.0;
        }
        if input_state.keys.is_down(VirtualKeyCode::S) {
            move_vec += Vec3::unit_z() * 1.0;
        }
        if input_state.keys.is_down(VirtualKeyCode::A) {
            move_vec += Vec3::unit_x() * -1.0;
        }
        if input_state.keys.is_down(VirtualKeyCode::D) {
            move_vec += Vec3::unit_x() * 1.0;
        }
        if input_state.keys.is_down(VirtualKeyCode::Q) {
            move_vec += Vec3::unit_y() * -1.0;
        }
        if input_state.keys.is_down(VirtualKeyCode::E) {
            move_vec += Vec3::unit_y() * 1.0;
        }

        move_vec *= 0.5;

        if input_state.keys.is_down(VirtualKeyCode::LControl) {
            move_vec *= 0.1;
        }

        if input_state.keys.is_down(VirtualKeyCode::LShift) {
            move_vec *= 10.0;
        }

        FirstPersonCameraInput {
            move_vec,
            yaw_delta,
            pitch_delta,
            dt: input_state.dt,
        }
    }
}
