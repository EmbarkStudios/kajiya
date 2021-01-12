mod asset;
mod camera;
mod image_cache;
mod input;
mod logging;
mod math;
mod render_client;
mod render_passes;
mod viewport;

use asset::mesh::*;
use camera::*;
use image_cache::*;
use input::*;
use math::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use render_client::BindlessImageHandle;
use slingshot::*;
use std::{collections::HashMap, sync::Arc};
use turbosloth::*;
use winit::{ElementState, Event, KeyboardInput, MouseButton, WindowBuilder, WindowEvent};

pub struct FrameState {
    pub camera_matrices: CameraMatrices,
    pub window_cfg: WindowConfig,
    pub input: InputState,
}

fn try_main() -> anyhow::Result<()> {
    logging::set_up_logging()?;

    let mut event_loop = winit::EventsLoop::new();

    let window_cfg = WindowConfig {
        width: 1280,
        height: 720,
    };

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("vicki")
            .with_dimensions(winit::dpi::LogicalSize::new(
                window_cfg.width as f64,
                window_cfg.height as f64,
            ))
            .build(&event_loop)
            .expect("window"),
    );

    let lazy_cache = LazyCache::create();

    let render_backend = RenderBackend::new(&*window, &window_cfg)?;
    let mut render_client = render_client::VickiRenderClient::new(&render_backend)?;
    let mut renderer = renderer::Renderer::new(render_backend)?;

    let mut last_error_text = None;

    #[allow(unused_mut)]
    let mut camera = camera::FirstPersonCamera::new(Vec3::new(0.0, 2.0, 10.0));

    let mut mouse_state: MouseState = Default::default();
    let mut keyboard: KeyboardState = Default::default();

    let mut keyboard_events: Vec<KeyboardInput> = Vec::new();
    let mut new_mouse_state: MouseState = Default::default();

    let mesh = LoadGltfScene {
        path: "assets/meshes2/rm_342/scene.gltf".into(),
        scale: 0.03,
    }
    .into_lazy();
    let mesh = smol::block_on(mesh.eval(&lazy_cache))?;

    let mut image_cache = ImageCache::new(lazy_cache.clone());
    let mut cached_image_to_bindless_handle: HashMap<usize, BindlessImageHandle> =
        Default::default();

    let mut mesh = pack_triangle_mesh(&mesh);
    {
        let mesh_map_gpu_ids: Vec<BindlessImageHandle> = mesh
            .maps
            .iter()
            .map(|map| {
                let img = image_cache.load_mesh_map(map).unwrap();
                match img {
                    ImageCacheResponse::Hit { id } => cached_image_to_bindless_handle[&id],
                    ImageCacheResponse::Miss { id, image } => {
                        let handle = render_client.add_image(image.as_ref());
                        cached_image_to_bindless_handle.insert(id, handle);
                        handle
                    }
                }
            })
            .collect();
        for mat in &mut mesh.materials {
            for m in &mut mat.maps {
                *m = mesh_map_gpu_ids[*m as usize].0;
            }
        }
    }

    render_client.add_mesh(mesh);
    render_client.build_ray_tracing_top_level_acceleration();

    let mut last_frame_instant = std::time::Instant::now();
    let mut running = true;
    while running {
        let mut events = Vec::new();
        event_loop.poll_events(|event| {
            events.push(event);
        });

        for event in events.into_iter() {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => running = false,
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
                _ => (),
            }
        }

        let now = std::time::Instant::now();
        let dt_duration = now - last_frame_instant;
        last_frame_instant = now;
        let dt = dt_duration.as_secs_f32();

        keyboard.update(std::mem::take(&mut keyboard_events), dt);
        mouse_state.update(&new_mouse_state);
        new_mouse_state = mouse_state.clone();

        let input_state = InputState {
            mouse: mouse_state,
            keys: keyboard.clone(),
            dt,
        };
        camera.update(&input_state);

        let frame_state = FrameState {
            camera_matrices: camera.calc_matrices(),
            window_cfg: window_cfg,
            input: input_state,
        };

        match renderer.prepare_frame(&mut render_client, &frame_state) {
            Ok(()) => {
                renderer.draw_frame(&mut render_client, &frame_state);
                last_error_text = None;
            }
            Err(e) => {
                let error_text = Some(format!("{:?}", e));
                if error_text != last_error_text {
                    println!("{}", error_text.as_ref().unwrap());
                    last_error_text = error_text;
                }
            }
        }
    }

    Ok(())
}

fn main() {
    /*panic::set_hook(Box::new(|panic_info| {
        println!("rust panic: {}", panic_info);
        loop {}
    }));*/

    if let Err(err) = try_main() {
        eprintln!("ERROR: {:?}", err);
        // std::thread::sleep(std::time::Duration::from_secs(1));
        loop {}
    }
}
