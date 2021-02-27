mod asset;
mod camera;
mod image_cache;
mod image_lut;
mod imgui_backend;
mod input;
mod logging;
mod lut_renderers;
mod math;
mod render_client;
mod render_passes;
mod renderers;
mod viewport;

use asset::{
    image::{CreatePlaceholderImage, LoadImage, RawRgba8Image},
    mesh::*,
};
use camera::*;
use image_cache::*;
use imgui::im_str;
use input::*;
use lut_renderers::*;
use math::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use parking_lot::Mutex;
use render_client::{BindlessImageHandle, RenderMode};
use slingshot::*;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    sync::Arc,
};
use turbosloth::*;
use winit::{ElementState, Event, KeyboardInput, MouseButton, WindowBuilder, WindowEvent};

use std::path::PathBuf;
use structopt::StructOpt;

use async_channel::unbounded;
use async_executor::Executor;
use easy_parallel::Parallel;

pub struct FrameState {
    pub camera_matrices: CameraMatrices,
    pub window_cfg: WindowConfig,
    pub input: InputState,
    pub sun_direction: Vec3,
}

#[derive(Debug, StructOpt)]
#[structopt(name = "vicki", about = "Toy rendering engine.")]
struct Opt {
    #[structopt(long, default_value = "1280")]
    width: u32,

    #[structopt(long, default_value = "720")]
    height: u32,

    #[structopt(long, parse(from_os_str))]
    scene: PathBuf,

    #[structopt(long, default_value = "0.1")]
    scale: f32,

    #[structopt(long)]
    no_vsync: bool,

    #[structopt(long)]
    no_debug: bool,
}

fn main() -> anyhow::Result<()> {
    logging::set_up_logging()?;
    let opt = Opt::from_args();

    let mut event_loop = winit::EventsLoop::new();

    let window_cfg = WindowConfig {
        width: opt.width,
        height: opt.height,
        vsync: !opt.no_vsync,
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

    let render_backend = RenderBackend::new(&*window, &window_cfg, !opt.no_debug)?;
    let mut render_client = render_client::VickiRenderClient::new(&render_backend)?;
    render_client.add_image_lut(BrdfFgLutComputer, 0);

    let mut renderer = renderer::Renderer::new(render_backend)?;
    let mut last_error_text = None;

    {
        let image = LoadImage::new("assets/images/bluenoise/256_256/LDR_RGBA_0.png")?.into_lazy();
        let blue_noise_img = smol::block_on(
            UploadGpuImage {
                image,
                params: TexParams {
                    gamma: TexGamma::Linear,
                    use_mips: false,
                },
                device: renderer.device().clone(),
            }
            .into_lazy()
            .eval(&lazy_cache),
        )
        .unwrap();

        render_client.add_image(blue_noise_img);
    }

    #[allow(unused_mut)]
    let mut camera =
        CameraConvergenceEnforcer::new(camera::FirstPersonCamera::new(Vec3::new(0.0, 2.0, 10.0)));

    let mut mouse_state: MouseState = Default::default();
    let mut keyboard: KeyboardState = Default::default();

    let mut keyboard_events: Vec<KeyboardInput> = Vec::new();
    let mut new_mouse_state: MouseState = Default::default();

    let mesh = LoadGltfScene {
        path: opt.scene,
        scale: opt.scale,
    }
    .into_lazy();

    let mesh = smol::block_on(mesh.eval(&lazy_cache))?;

    let mut material_map_to_bindless_handlee: HashMap<MeshMaterialMap, BindlessImageHandle> =
        Default::default();

    let mesh_images: Vec<Lazy<Image>> = mesh
        .maps
        .iter()
        .map(|map| {
            let (image, params) = match map {
                MeshMaterialMap::Asset { path, params } => {
                    (LoadImage::new(&path).unwrap().into_lazy(), *params)
                }
                MeshMaterialMap::Placeholder(values) => (
                    CreatePlaceholderImage::new(*values).into_lazy(),
                    TexParams {
                        gamma: crate::asset::mesh::TexGamma::Linear,
                        use_mips: false,
                    },
                ),
            };

            UploadGpuImage {
                image,
                params,
                device: renderer.device().clone(),
            }
            .into_lazy()
        })
        .collect();

    /*{
        let ex = Executor::new();
        let (signal, shutdown) = unbounded::<()>();

        Parallel::new()
            // Run four executor threads.
            .each(0..16, |_| {
                futures_lite::future::block_on(ex.run(shutdown.recv()))
            })
            // Run the main future on the current thread.
            .finish(|| {
                futures_lite::future::block_on(async {
                    let stream = stream::iter(mesh_images.into_iter()).then(|img| {
                        ex.spawn({
                            let lazy_cache = lazy_cache.clone();
                            async move {
                                loop {}
                                img.eval(&lazy_cache).await
                            }
                        })
                    });
                    let images = stream.collect::<Vec<_>>().await;
                    drop(signal);
                })
            });
    }*/

    let loaded_images = mesh_images
        .iter()
        .cloned()
        .map(|img| smol::spawn(img.eval(&lazy_cache)));
    let loaded_images = smol::block_on(futures::future::try_join_all(loaded_images))
        .expect("Failed to load mesh images");

    let mut mesh = pack_triangle_mesh(&mesh);
    {
        let mesh_map_gpu_ids: Vec<BindlessImageHandle> = mesh
            .maps
            .iter()
            .zip(loaded_images.iter())
            .map(|(map, img)| {
                *material_map_to_bindless_handlee
                    .entry(map.clone())
                    .or_insert_with(|| render_client.add_image(img.clone()))
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

    let mut imgui = imgui::Context::create();
    let mut imgui_backend =
        imgui_backend::ImGuiBackend::new(renderer.device().clone(), &window, &mut imgui);
    imgui_backend.create_graphics_resources([window_cfg.width, window_cfg.height]);
    let imgui_backend = Arc::new(Mutex::new(imgui_backend));

    let mut light_theta = -4.54;
    let mut light_phi = 1.48;

    let mut last_frame_instant = std::time::Instant::now();
    let mut running = true;
    while running {
        let mut events = Vec::new();
        event_loop.poll_events(|event| {
            imgui_backend
                .lock()
                .handle_event(window.as_ref(), &mut imgui, &event);
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

        if keyboard.was_just_pressed(VirtualKeyCode::Space) {
            render_client.render_mode = match render_client.render_mode {
                RenderMode::Standard => RenderMode::Reference,
                RenderMode::Reference => RenderMode::Standard,
            };
        }

        if (mouse_state.button_mask & 1) != 0 {
            light_theta +=
                (mouse_state.delta.x / window_cfg.width as f32) * std::f32::consts::PI * -2.0;
            light_phi +=
                (mouse_state.delta.y / window_cfg.height as f32) * std::f32::consts::PI * 0.5;
        }

        // Reset accumulation of the path tracer whenever the camera moves
        if !camera.is_converged() || keyboard.was_just_pressed(VirtualKeyCode::Back) {
            render_client.reset_reference_accumulation = true;
        }

        let frame_state = FrameState {
            camera_matrices: camera.calc_matrices(),
            window_cfg: window_cfg,
            input: input_state,
            //sun_direction: Vec3::new(-0.8, 0.3, 1.0).normalize(),
            sun_direction: spherical_to_cartesian(light_theta, light_phi),
        };

        let (ui_draw_data, imgui_target_image) = {
            let mut imgui_backend = imgui_backend.lock();
            let ui = imgui_backend.prepare_frame(&window, &mut imgui, dt);

            if ui
                .collapsing_header(im_str!("GPU passes"))
                .default_open(true)
                .build()
            {
                let gpu_stats = gpu_profiler::get_stats();
                ui.text(format!("CPU frame time: {:.3}ms", dt * 1000.0));

                for (name, ms) in gpu_stats.get_ordered_name_ms() {
                    ui.text(format!("{}: {:.3}ms", name, ms));
                }
            }

            if ui
                .collapsing_header(im_str!("csgi"))
                .default_open(true)
                .build()
            {
                ui.drag_int(
                    im_str!("Trace subdivision"),
                    &mut render_client.csgi.trace_subdiv,
                )
                .min(0)
                .max(5)
                .build();

                ui.drag_int(
                    im_str!("Neighbors per frame"),
                    &mut render_client.csgi.neighbors_per_frame,
                )
                .min(1)
                .max(9)
                .build();
            }

            imgui_backend.prepare_render(&ui, &window);
            (ui.render(), imgui_backend.get_target_image().unwrap())
        };

        let ui_draw_data: &'static imgui::DrawData = unsafe { std::mem::transmute(ui_draw_data) };
        let imgui_backend = imgui_backend.clone();
        let gui_extent = [frame_state.window_cfg.width, frame_state.window_cfg.height];

        render_client.ui_frame = Some((
            Box::new(move |cb| {
                imgui_backend
                    .lock()
                    .render(gui_extent, ui_draw_data, cb)
                    .expect("ui.render");
            }),
            imgui_target_image,
        ));

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

fn spherical_to_cartesian(theta: f32, phi: f32) -> Vec3 {
    let x = phi.sin() * theta.cos();
    let y = phi.cos();
    let z = phi.sin() * theta.sin();
    Vec3::new(x, y, z)
}
