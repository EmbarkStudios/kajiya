#![allow(dead_code)]

use glam::Vec2;
use std::collections::HashMap;
pub use winit::event::{ElementState, KeyboardInput, VirtualKeyCode};

#[derive(Clone)]
pub struct KeyState {
    pub ticks: u32,
    pub seconds: f32,
}

#[derive(Default, Clone)]
pub struct KeyboardState {
    keys_down: HashMap<VirtualKeyCode, KeyState>,
    events: Vec<KeyboardInput>,
}

impl KeyboardState {
    pub fn is_down(&self, key: VirtualKeyCode) -> bool {
        self.get_down(key).is_some()
    }

    pub fn was_just_pressed(&self, key: VirtualKeyCode) -> bool {
        self.get_down(key).map(|s| s.ticks == 1).unwrap_or_default()
    }

    pub fn get_down(&self, key: VirtualKeyCode) -> Option<&KeyState> {
        self.keys_down.get(&key)
    }

    pub fn iter_events(&self) -> impl Iterator<Item = &KeyboardInput> {
        self.events.iter()
    }

    pub fn update(&mut self, events: Vec<KeyboardInput>, dt: f32) {
        self.events = events;

        for event in &self.events {
            if let Some(vk) = event.virtual_keycode {
                if event.state == ElementState::Pressed {
                    self.keys_down.entry(vk).or_insert(KeyState {
                        ticks: 0,
                        seconds: 0.0,
                    });
                } else {
                    self.keys_down.remove(&vk);
                }
            }
        }

        for ks in self.keys_down.values_mut() {
            ks.ticks += 1;
            ks.seconds += dt;
        }
    }
}

#[derive(Clone, Copy)]
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
