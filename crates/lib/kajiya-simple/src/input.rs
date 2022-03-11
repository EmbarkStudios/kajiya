#![allow(dead_code)]

use glam::Vec2;
use std::collections::HashMap;
pub use winit::event::{ElementState, KeyboardInput, VirtualKeyCode};
use winit::{
    dpi::PhysicalPosition,
    event::{MouseScrollDelta, WindowEvent},
};

#[derive(Clone)]
pub struct KeyState {
    pub ticks: u32,
}

#[derive(Default, Clone)]
pub struct KeyboardState {
    keys_down: HashMap<VirtualKeyCode, KeyState>,
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

    pub fn update(&mut self, events: &[WindowEvent]) {
        for event in events {
            if let WindowEvent::KeyboardInput { input, .. } = event {
                if let Some(vk) = input.virtual_keycode {
                    if input.state == ElementState::Pressed {
                        self.keys_down.entry(vk).or_insert(KeyState { ticks: 0 });
                    } else {
                        self.keys_down.remove(&vk);
                    }
                }
            }
        }

        for ks in self.keys_down.values_mut() {
            ks.ticks += 1;
        }
    }
}

#[derive(Clone, Copy)]
pub struct MouseState {
    pub physical_position: PhysicalPosition<f64>,
    pub delta: Vec2,
    pub buttons_held: u32,
    pub buttons_pressed: u32,
    pub buttons_released: u32,
    pub wheel_delta: f32,
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            physical_position: PhysicalPosition { x: 0.0, y: 0.0 },
            delta: Vec2::ZERO,
            buttons_held: 0,
            buttons_pressed: 0,
            buttons_released: 0,
            wheel_delta: 0.0,
        }
    }
}

impl MouseState {
    pub fn update(&mut self, events: &[WindowEvent]) {
        let prev_physical_position = self.physical_position;
        self.buttons_pressed = 0;
        self.buttons_released = 0;
        self.wheel_delta = 0.0;

        for event in events {
            match event {
                WindowEvent::CursorMoved { position, .. } => {
                    self.physical_position = *position;
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let button_id = match button {
                        winit::event::MouseButton::Left => 0,
                        winit::event::MouseButton::Middle => 1,
                        winit::event::MouseButton::Right => 2,
                        _ => 0,
                    };

                    if let ElementState::Pressed = state {
                        self.buttons_held |= 1 << button_id;
                        self.buttons_pressed |= 1 << button_id;
                    } else {
                        self.buttons_held &= !(1 << button_id);
                        self.buttons_released |= 1 << button_id;
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    if let MouseScrollDelta::LineDelta(_, delta_y) = delta {
                        self.wheel_delta = *delta_y * 24.0;
                    }
                }
                _ => (),
            }
        }

        self.delta = Vec2::new(
            (self.physical_position.x - prev_physical_position.x) as f32,
            (self.physical_position.y - prev_physical_position.y) as f32,
        );
    }
}

pub type InputAxis = &'static str;

pub struct KeyMap {
    axis: InputAxis,
    multiplier: f32,
    activation_time: f32,
}

impl KeyMap {
    pub fn new(axis: InputAxis, multiplier: f32) -> Self {
        Self {
            axis,
            multiplier,
            activation_time: 0.15,
        }
    }

    pub fn activation_time(mut self, value: f32) -> Self {
        self.activation_time = value;
        self
    }
}

struct KeyMapState {
    map: KeyMap,
    activation: f32,
}

pub struct KeyboardMap {
    bindings: Vec<(VirtualKeyCode, KeyMapState)>,
}

impl Default for KeyboardMap {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyboardMap {
    pub fn new() -> Self {
        Self {
            bindings: Default::default(),
        }
    }

    pub fn bind(mut self, key: VirtualKeyCode, map: KeyMap) -> Self {
        self.bindings.push((
            key,
            KeyMapState {
                map,
                activation: 0.0,
            },
        ));
        self
    }

    pub fn map(&mut self, keyboard: &KeyboardState, dt: f32) -> HashMap<InputAxis, f32> {
        let mut result: HashMap<InputAxis, f32> = HashMap::new();

        for (vk, s) in &mut self.bindings {
            #[allow(clippy::collapsible_else_if)]
            if s.map.activation_time > 1e-10 {
                let change = if keyboard.is_down(*vk) { dt } else { -dt };
                s.activation = (s.activation + change / s.map.activation_time).clamp(0.0, 1.0);
            } else {
                if keyboard.is_down(*vk) {
                    s.activation = 1.0;
                } else {
                    s.activation = 0.0;
                }
            }

            *result.entry(s.map.axis).or_default() += s.activation * s.map.multiplier;
        }

        for value in result.values_mut() {
            *value = value.clamp(-1.0, 1.0);
        }

        result
    }
}
