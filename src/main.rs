mod backend;
mod bytes;
mod camera;
mod chunky_list;
mod dynamic_constants;
mod file;
mod logging;
mod math;
mod mesh;
mod pipeline_cache;
mod renderer;
mod shader_compiler;
mod viewport;

use backend::RenderBackend;
use camera::*;

use glam::Vec3;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
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

    let mut backend = RenderBackend::new(&*window, &window_cfg)?;
    let mut renderer = renderer::Renderer::new(&backend)?;
    let mut last_error_text = None;

    #[allow(unused_mut)]
    let mut camera = camera::FirstPersonCamera::new(Vec3::new(0.0, 2.0, 10.0));

    event_loop.run(move |event, _, control_flow| {
        // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
        // dispatched any events. This is ideal for games and similar applications.
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
                // Application update code.
                window.request_redraw();
            }
            Event::RedrawRequested(_) => match renderer.prepare_frame(&backend) {
                Ok(()) => {
                    renderer.draw_frame(&mut backend, camera.calc_matrices());
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
