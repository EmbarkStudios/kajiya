use anyhow::{anyhow, Context};
use kajiya_simple::VirtualKeyCode::*;
use kajiya_simple::{KeyMap, KeyboardMap, VirtualKeyCode};
use serde::{Deserialize, Serialize};
use std::fs::{canonicalize, File};
use std::io::Read;
use std::path::PathBuf;
use toml::from_str;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct KeymapConfig {
    pub movement: Movement,
    pub ui: Ui,
    pub sequencer: Sequencer,
    pub rendering: Rendering,
    pub misc: Misc,
}

impl KeymapConfig {
    pub(crate) fn load(path: &Option<PathBuf>) -> anyhow::Result<Self> {
        let path = path.clone().unwrap_or("keymap.toml".into());
        let path = canonicalize(path).with_context(|| {
            "Failed to find keymap.toml. Make sure it is in the same directory as the executable."
        })?;

        let mut keymap_file = File::open(path).with_context(|| "Failed to open keymap.toml")?;

        let mut buffer = String::new();
        keymap_file
            .read_to_string(&mut buffer)
            .with_context(|| "Failed to read keymap.toml")?;

        // Don't use anyhow context here because it doesn't show the parsing error.
        let keymap = from_str(&buffer)
            .map_err(|e| anyhow!("Failed to parse keymap.toml: {}", e.to_string()))?;

        Ok(keymap)
    }
}

impl From<Movement> for KeyboardMap {
    fn from(val: Movement) -> Self {
        KeyboardMap::new()
            .bind(val.forward, KeyMap::new("move_fwd", 1.0))
            .bind(val.backward, KeyMap::new("move_fwd", -1.0))
            .bind(val.right, KeyMap::new("move_right", 1.0))
            .bind(val.left, KeyMap::new("move_right", -1.0))
            .bind(val.up, KeyMap::new("move_up", 1.0))
            .bind(val.down, KeyMap::new("move_up", -1.0))
            .bind(val.boost, KeyMap::new("boost", 1.0).activation_time(0.25))
            .bind(val.slow, KeyMap::new("boost", -1.0).activation_time(0.5))
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Movement {
    forward: VirtualKeyCode,
    backward: VirtualKeyCode,
    left: VirtualKeyCode,
    right: VirtualKeyCode,
    up: VirtualKeyCode,
    down: VirtualKeyCode,
    boost: VirtualKeyCode,
    slow: VirtualKeyCode,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Ui {
    pub toggle: VirtualKeyCode,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Sequencer {
    pub add_keyframe: VirtualKeyCode,
    pub play: VirtualKeyCode,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Rendering {
    pub switch_to_reference_path_tracing: VirtualKeyCode,
    pub reset_path_tracer: VirtualKeyCode,
    pub light_enable_emissive: VirtualKeyCode,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Misc {
    pub print_camera_transform: VirtualKeyCode,
}

impl Default for Movement {
    fn default() -> Self {
        Self {
            forward: W,
            backward: S,
            left: A,
            right: D,
            up: E,
            down: Q,
            boost: LShift,
            slow: LControl,
        }
    }
}

impl Default for Ui {
    fn default() -> Self {
        Self { toggle: Tab }
    }
}

impl Default for Sequencer {
    fn default() -> Self {
        Self {
            add_keyframe: K,
            play: P,
        }
    }
}

impl Default for Rendering {
    fn default() -> Self {
        Self {
            switch_to_reference_path_tracing: Space,
            reset_path_tracer: Back,
            light_enable_emissive: L,
        }
    }
}

impl Default for Misc {
    fn default() -> Self {
        Self {
            print_camera_transform: C,
        }
    }
}
