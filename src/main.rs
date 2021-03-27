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

use anyhow::Context;
use asset::{image::LoadImage, mesh::*};
use camera::*;
use image_cache::*;
use imgui::im_str;
use input::*;
use lut_renderers::*;
use math::*;

use kajiya_backend::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use memmap2::MmapOptions;
use parking_lot::Mutex;
use render_client::{RenderDebugMode, RenderMode};
use std::{collections::HashMap, fs::File, sync::Arc};
use structopt::StructOpt;
use turbosloth::*;
use winit::{
    event::{ElementState, Event, KeyboardInput, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub struct FrameState {
    pub camera_matrices: CameraMatrices,
    pub window_cfg: WindowConfig,
    //pub swapchain_extent: [u32; 2],
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

    #[structopt(long)]
    no_vsync: bool,

    #[structopt(long)]
    no_window_decorations: bool,

    #[structopt(long)]
    no_debug: bool,
}

#[derive(serde::Deserialize)]
struct SceneDesc {
    instances: Vec<SceneInstanceDesc>,
}

#[derive(serde::Deserialize)]
struct SceneInstanceDesc {
    position: [f32; 3],
    mesh: String,
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

    let scene_file = format!("assets/scenes/{}.ron", opt.scene);
    let scene_desc: SceneDesc = ron::de::from_reader(
        File::open(&scene_file).with_context(|| format!("Opening scene file {}", scene_file))?,
    )?;

    let event_loop = EventLoop::new();

    let window_cfg = WindowConfig {
        width: opt.width,
        height: opt.height,
        vsync: !opt.no_vsync,
    };

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("kajiya")
            .with_resizable(false)
            .with_decorations(!opt.no_window_decorations)
            .with_inner_size(winit::dpi::LogicalSize::new(
                window_cfg.width as f64,
                window_cfg.height as f64,
            ))
            .build(&event_loop)
            .expect("window"),
    );

    let lazy_cache = LazyCache::create();

    let render_backend = RenderBackend::new(&*window, &window_cfg, !opt.no_debug)?;
    let mut render_client = render_client::KajiyaRenderClient::new(&render_backend)?;

    // BINDLESS_LUT_BRDF_FG
    render_client.add_image_lut(BrdfFgLutComputer, 0);

    let mut renderer = kajiya_rg::renderer::Renderer::new(render_backend)?;
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

        let handle = render_client.add_image(blue_noise_img);

        // BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0
        assert_eq!(handle.0, 1);
    }

    let mut camera = camera::FirstPersonCamera::new(Vec3::new(0.0, 1.0, 8.0));
    //camera.fov = 65.0;
    //camera.look_smoothness = 20.0;
    //camera.move_smoothness = 20.0;

    camera.look_smoothness = 3.0;
    camera.move_smoothness = 3.0;

    // Mitsuba match
    /*let mut camera = camera::FirstPersonCamera::new(Vec3::new(-2.0, 4.0, 8.0));
    camera.fov = 35.0 * 9.0 / 16.0;
    camera.look_at(Vec3::new(0.0, 0.75, 0.0));*/

    camera.aspect = window_cfg.width as f32 / window_cfg.height as f32;

    #[allow(unused_mut)]
    let mut camera = CameraConvergenceEnforcer::new(camera);
    camera.convergence_sensitivity = 0.0;

    let mut mouse_state: MouseState = Default::default();
    let mut keyboard: KeyboardState = Default::default();

    let mut keyboard_events: Vec<KeyboardInput> = Vec::new();
    let mut new_mouse_state: MouseState = Default::default();

    for instance in scene_desc.instances {
        let mesh = mmapped_asset::<PackedTriMesh::Flat>(&format!("baked/{}.mesh", instance.mesh))?;
        let mesh = render_client.add_mesh(mesh);
        render_client.add_instance(mesh, instance.position.into());
    }

    /*let car_mesh = mmapped_asset::<PackedTriMesh::Flat>("baked/336_lrm.mesh")?;
    let car_mesh = render_client.add_mesh(car_mesh);
    let mut car_pos = Vec3::unit_y() * -0.01;
    let car_inst = render_client.add_instance(car_mesh, car_pos);*/

    render_client.build_ray_tracing_top_level_acceleration();

    let mut imgui = imgui::Context::create();
    let mut imgui_backend =
        imgui_backend::ImGuiBackend::new(renderer.device().clone(), &window, &mut imgui);
    imgui_backend
        .create_graphics_resources([window.inner_size().width, window.inner_size().height]);
    let imgui_backend = Arc::new(Mutex::new(imgui_backend));
    let mut show_gui = true;

    let mut light_theta = -4.54;
    let mut light_phi = 1.48;
    let mut sun_direction_interp = spherical_to_cartesian(light_theta, light_phi);

    let mut last_frame_instant = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        imgui_backend
            .lock()
            .handle_event(window.as_ref(), &mut imgui, &event);

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

                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
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

                render_client.store_prev_mesh_transforms();

                // Reset accumulation of the path tracer whenever the camera moves
                if (!camera.is_converged() || keyboard.was_just_pressed(VirtualKeyCode::Back))
                    && render_client.render_mode == RenderMode::Reference
                {
                    render_client.reset_reference_accumulation = true;
                }

                /*if keyboard.is_down(VirtualKeyCode::Z) {
                    car_pos.x += mouse_state.delta.x / 100.0;
                    render_client.set_instance_transform(car_inst, car_pos);
                }*/

                if keyboard.was_just_pressed(VirtualKeyCode::Space) {
                    match render_client.render_mode {
                        RenderMode::Standard => {
                            camera.convergence_sensitivity = 1.0;
                            render_client.render_mode = RenderMode::Reference;
                        }
                        RenderMode::Reference => {
                            camera.convergence_sensitivity = 0.0;
                            render_client.render_mode = RenderMode::Standard;
                        }
                    };
                }

                if (mouse_state.button_mask & 1) != 0 {
                    light_theta +=
                        (mouse_state.delta.x / window_cfg.width as f32) * -std::f32::consts::TAU;
                    light_phi +=
                        (mouse_state.delta.y / window_cfg.height as f32) * std::f32::consts::PI;
                }

                let sun_direction = spherical_to_cartesian(light_theta, light_phi);
                sun_direction_interp =
                    Vec3::lerp(sun_direction_interp, sun_direction, 0.1).normalize();

                let frame_state = FrameState {
                    camera_matrices: camera.calc_matrices(),
                    window_cfg,
                    //swapchain_extent: [window.inner_size().width, window.inner_size().height],
                    input: InputState {
                        mouse: mouse_state,
                        keys: keyboard.clone(),
                        dt,
                    },
                    //sun_direction: (Vec3::new(-6.0, 4.0, -6.0)).normalize(),
                    sun_direction: sun_direction_interp,
                };

                if keyboard.was_just_pressed(VirtualKeyCode::Tab) {
                    show_gui = !show_gui;
                }

                if show_gui {
                    let (ui_draw_data, imgui_target_image) = {
                        let mut imgui_backend = imgui_backend.lock();
                        let ui = imgui_backend.prepare_frame(&window, &mut imgui, dt);

                        if imgui::CollapsingHeader::new(im_str!("GPU passes"))
                            .default_open(true)
                            .build(&ui)
                        {
                            let gpu_stats = gpu_profiler::get_stats();
                            ui.text(format!("CPU frame time: {:.3}ms", dt * 1000.0));

                            let mut sum = 0.0;
                            for (name, ms) in gpu_stats.get_ordered_name_ms() {
                                ui.text(format!("{}: {:.3}ms", name, ms));
                                sum += ms;
                            }

                            ui.text(format!("total: {:.3}ms", sum));
                        }

                        if imgui::CollapsingHeader::new(im_str!("csgi"))
                            .default_open(true)
                            .build(&ui)
                        {
                            imgui::Drag::<i32>::new(im_str!("Trace subdivision"))
                                .range(0..=5)
                                .build(&ui, &mut render_client.csgi.trace_subdiv);

                            imgui::Drag::<i32>::new(im_str!("Neighbors per frame"))
                                .range(1..=9)
                                .build(&ui, &mut render_client.csgi.neighbors_per_frame);
                        }

                        if imgui::CollapsingHeader::new(im_str!("debug"))
                            .default_open(true)
                            .build(&ui)
                        {
                            if ui.radio_button_bool(
                                im_str!("none"),
                                render_client.debug_mode == RenderDebugMode::None,
                            ) {
                                render_client.debug_mode = RenderDebugMode::None;
                            }

                            if ui.radio_button_bool(
                                im_str!("csgi voxel grid"),
                                render_client.debug_mode == RenderDebugMode::CsgiVoxelGrid,
                            ) {
                                render_client.debug_mode = RenderDebugMode::CsgiVoxelGrid;
                            }
                        }

                        imgui_backend.prepare_render(&ui, &window);
                        (ui.render(), imgui_backend.get_target_image().unwrap())
                    };

                    let ui_draw_data: &'static imgui::DrawData =
                        unsafe { std::mem::transmute(ui_draw_data) };
                    let imgui_backend = imgui_backend.clone();
                    let gui_extent = [window.inner_size().width, window.inner_size().height];

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
            _ => (),
        }
    })
}

fn spherical_to_cartesian(theta: f32, phi: f32) -> Vec3 {
    let x = phi.sin() * theta.cos();
    let y = phi.cos();
    let z = phi.sin() * theta.sin();
    Vec3::new(x, y, z)
}
