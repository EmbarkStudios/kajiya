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

use asset::{image::LoadImage, mesh::*};
use camera::*;
use image_cache::*;
use imgui::im_str;
use input::*;
use lut_renderers::*;
use math::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use memmap2::MmapOptions;
use parking_lot::Mutex;
use render_client::{RenderDebugMode, RenderMode};
use slingshot::*;
use std::{collections::HashMap, fs::File, sync::Arc};
use turbosloth::*;
use winit::{ElementState, Event, KeyboardInput, MouseButton, WindowBuilder, WindowEvent};

use structopt::StructOpt;

pub struct FrameState {
    pub camera_matrices: CameraMatrices,
    pub window_cfg: WindowConfig,
    pub input: InputState,
    pub sun_direction: Vec3,
}

#[derive(Debug, StructOpt)]
#[structopt(name = "kajiya", about = "Toy rendering engine.")]
struct Opt {
    #[structopt(long, default_value = "1280")]
    width: u32,

    #[structopt(long, default_value = "720")]
    height: u32,

    //#[structopt(long, parse(from_os_str))]
    //scene: PathBuf,
    #[structopt(long)]
    scene: String,

    #[structopt(long, default_value = "0.0")]
    y_offset: f32,

    #[structopt(long)]
    no_vsync: bool,

    #[structopt(long)]
    no_debug: bool,
}

lazy_static::lazy_static! {
    static ref ASSET_MMAPS: Mutex<HashMap<String, memmap2::Mmap>> = Mutex::new(HashMap::new());
}

pub fn mmapped_asset<T>(path: &str) -> anyhow::Result<&'static T> {
    let mut mmaps = ASSET_MMAPS.lock();
    let data: &[u8] = mmaps.entry(path.to_owned()).or_insert_with(|| {
        let file = File::open(path).unwrap();
        unsafe { MmapOptions::new().map(&file).unwrap() }
    });
    let asset: &T = unsafe { (data.as_ptr() as *const T).as_ref() }.unwrap();
    Ok(asset)
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
            .with_title("kajiya")
            .with_decorations(false)
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

    let mut camera = camera::FirstPersonCamera::new(Vec3::new(0.0, 0.0, 8.0));
    //camera.fov = 65.0;

    // Mitsuba match
    /*let mut camera = camera::FirstPersonCamera::new(Vec3::new(-2.0, 4.0, 8.0));
    camera.fov = 35.0 * 9.0 / 16.0;
    camera.look_at(Vec3::new(0.0, 0.75, 0.0));*/

    camera.aspect = window_cfg.width as f32 / window_cfg.height as f32;

    #[allow(unused_mut)]
    let mut camera = CameraConvergenceEnforcer::new(camera);

    let mut mouse_state: MouseState = Default::default();
    let mut keyboard: KeyboardState = Default::default();

    let mut keyboard_events: Vec<KeyboardInput> = Vec::new();
    let mut new_mouse_state: MouseState = Default::default();

    /*let mesh = mmapped_asset::<PackedTriMesh::Flat>(&format!("baked/{}.mesh", opt.scene))?;
    let mesh = render_client.add_mesh(mesh);
    render_client.add_instance(mesh, Vec3::new(-1.5, 0.0, -2.5));
    render_client.add_instance(mesh, Vec3::new(1.5, 0.0, -2.5));

    let mesh = mmapped_asset::<PackedTriMesh::Flat>("baked/336_lrm.mesh")?;
    let mesh = render_client.add_mesh(mesh);
    render_client.add_instance(mesh, Vec3::new(-1.5, 0.0, 2.5));
    render_client.add_instance(mesh, Vec3::new(1.5, 0.0, 2.5));*/

    /*let mesh = mmapped_asset::<PackedTriMesh::Flat>("baked/floor.mesh")?;
    let mesh = render_client.add_mesh(mesh);
    render_client.add_instance(mesh, Vec3::new(0.0, 0.0, 0.0));*/

    let mesh = mmapped_asset::<PackedTriMesh::Flat>(&format!("baked/{}.mesh", opt.scene))?;
    let mesh = render_client.add_mesh(mesh);
    render_client.add_instance(mesh, Vec3::new(0.0, opt.y_offset, 0.0));
    /*render_client.add_instance(mesh, Vec3::new(-2.0, opt.y_offset, -2.5));
    render_client.add_instance(mesh, Vec3::new(2.0, opt.y_offset, -2.5));
    render_client.add_instance(mesh, Vec3::new(-2.0, opt.y_offset, 2.5));
    render_client.add_instance(mesh, Vec3::new(2.0, opt.y_offset, 2.5));*/

    render_client.build_ray_tracing_top_level_acceleration();

    let mut imgui = imgui::Context::create();
    let mut imgui_backend =
        imgui_backend::ImGuiBackend::new(renderer.device().clone(), &window, &mut imgui);
    imgui_backend.create_graphics_resources([window_cfg.width, window_cfg.height]);
    let imgui_backend = Arc::new(Mutex::new(imgui_backend));
    let mut show_gui = true;

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
            #[allow(clippy::single_match)]
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
        new_mouse_state = mouse_state;

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
            window_cfg,
            input: input_state,
            //sun_direction: (Vec3::new(-6.0, 4.0, -6.0)).normalize(),
            sun_direction: spherical_to_cartesian(light_theta, light_phi),
        };

        if keyboard.was_just_pressed(VirtualKeyCode::Tab) {
            show_gui = !show_gui;
        }

        if show_gui {
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

                if ui
                    .collapsing_header(im_str!("debug"))
                    .default_open(true)
                    .build()
                {
                    if ui.radio_button_bool(
                        im_str!("none"),
                        render_client.debug_mode == RenderDebugMode::None,
                    ) {
                        render_client.debug_mode = RenderDebugMode::None;
                    }

                    if ui.radio_button_bool(
                        im_str!("csgi2 voxel grid"),
                        render_client.debug_mode == RenderDebugMode::Csgi2VoxelGrid,
                    ) {
                        render_client.debug_mode = RenderDebugMode::Csgi2VoxelGrid;
                    }
                }

                imgui_backend.prepare_render(&ui, &window);
                (ui.render(), imgui_backend.get_target_image().unwrap())
            };

            let ui_draw_data: &'static imgui::DrawData =
                unsafe { std::mem::transmute(ui_draw_data) };
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
        }

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
