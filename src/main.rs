mod backend;
mod bytes;
mod camera;
mod chunky_list;
mod dynamic_constants;
mod file;
mod input;
mod logging;
mod math;
mod mesh;
mod pipeline_cache;
mod renderer;
mod shader_compiler;
mod viewport;

use backend::RenderBackend;
use camera::*;
use input::*;
use math::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;
use winit::{
    event::{ElementState, Event, KeyboardInput, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Copy, Clone)]
pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
}

pub struct FrameState {
    pub camera_matrices: CameraMatrices,
    pub window_cfg: WindowConfig,
    pub input: InputState,
}

fn try_main() -> anyhow::Result<()> {
    logging::set_up_logging()?;

    let event_loop = EventLoop::new();

    let window_cfg = WindowConfig {
        width: 1280,
        height: 720,
    };

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("vicki")
            .with_inner_size(winit::dpi::LogicalSize::new(
                window_cfg.width as f64,
                window_cfg.height as f64,
            ))
            .build(&event_loop)
            .expect("window"),
    );

    let mut renderer = renderer::Renderer::new(RenderBackend::new(&*window, &window_cfg)?)?;
    let mut last_error_text = None;

    #[allow(unused_mut)]
    let mut camera = camera::FirstPersonCamera::new(Vec3::new(0.0, 2.0, 10.0));

    let mut mouse_state: MouseState = Default::default();
    let mut keyboard: KeyboardState = Default::default();

    let mut keyboard_events: Vec<KeyboardInput> = Vec::new();
    let mut new_mouse_state: MouseState = Default::default();
    let mut last_frame_instant = std::time::Instant::now();
    let mut dt = 1.0 / 60.0;

    event_loop.run(move |event, _, control_flow| {
        // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
        // dispatched any events. This is ideal for games and similar applications.
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => {
                    keyboard_events.push(input);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    new_mouse_state.pos = Vec2::new(position.x as f32, position.y as f32);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let button_id = match button {
                        MouseButton::Left => 0,
                        MouseButton::Middle => 1,
                        MouseButton::Right => 2,
                        _ => 0,
                    };

                    if let ElementState::Pressed = state {
                        new_mouse_state.button_mask |= 1 << button_id;
                    } else {
                        new_mouse_state.button_mask &= !(1 << button_id);
                    }
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                // Application update code.

                let now = std::time::Instant::now();
                let dt_duration = now - last_frame_instant;
                last_frame_instant = now;
                dt = dt_duration.as_secs_f32();

                keyboard.update(std::mem::take(&mut keyboard_events), dt);
                mouse_state.update(&new_mouse_state);
                new_mouse_state = mouse_state.clone();

                let input_state = InputState {
                    mouse: mouse_state,
                    keys: keyboard.clone(),
                    dt,
                };
                camera.update(&input_state);

                window.request_redraw();
            }
            Event::RedrawRequested(_) => match renderer.prepare_frame() {
                Ok(()) => {
                    renderer.draw_frame(FrameState {
                        camera_matrices: camera.calc_matrices(),
                        window_cfg: window_cfg,
                        input: InputState {
                            mouse: mouse_state,
                            keys: keyboard.clone(),
                            dt,
                        },
                    });
                    last_error_text = None;
                }
                Err(e) => {
                    let error_text = Some(format!("{:?}", e));
                    if error_text != last_error_text {
                        println!("{}", error_text.as_ref().unwrap());
                        last_error_text = error_text;
                    }
                }
            },
            _ => (),
        }
    })
}

fn main() {
    if let Err(err) = try_main() {
        eprintln!("ERROR: {:?}", err);
    }
}
