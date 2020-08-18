use crate::{keyboard::KeyboardState, math::*};

#[derive(Clone)]
pub struct MouseState {
    pub pos: Vec2,
    pub delta: Vec2,
    pub button_mask: u32,
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            pos: Vec2::zero(),
            delta: Vec2::zero(),
            button_mask: 0,
        }
    }
}

impl MouseState {
    pub fn update(&mut self, new_state: &MouseState) {
        self.delta = new_state.pos - self.pos;
        self.pos = new_state.pos;
        self.button_mask = new_state.button_mask;
    }
}

pub struct FrameState<'a> {
    pub mouse: &'a MouseState,
    pub keys: &'a KeyboardState,
    pub dt: f32,
}
