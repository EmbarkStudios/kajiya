mod backend;
mod bytes;
mod file;
mod logging;
mod mesh;
mod renderer;
mod shader_compiler;

use backend::RenderBackend;

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
            Event::RedrawRequested(_) => {
                renderer.draw_frame(&mut backend);
            }
            _ => (),
        }
    })
}

fn main() {
    if let Err(err) = try_main() {
        eprintln!("ERROR: {:?}", err);
    }
}
