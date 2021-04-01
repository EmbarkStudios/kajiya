mod camera_input;
mod imgui_backend;
mod input;

use anyhow::Context;

use input::*;
use kajiya::{
    asset::mesh::*,
    backend::{vulkan::RenderBackendConfig, *},
    camera::*,
    frame_state::FrameState,
    math::*,
    mmap::mmapped_asset,
    render_client::KajiyaRenderClient,
    world_renderer::{RenderDebugMode, RenderMode},
};

use imgui::im_str;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use parking_lot::Mutex;
use std::{fs::File, sync::Arc};
use structopt::StructOpt;
use turbosloth::*;

use winit::{
    event::{ElementState, Event, KeyboardInput, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Debug, StructOpt)]
#[structopt(name = "view", about = "Kajiya scene viewer.")]
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

fn main() -> anyhow::Result<()> {
    kajiya::logging::set_up_logging()?;
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

    let swapchain_extent = [window.inner_size().width, window.inner_size().height];
    let lazy_cache = LazyCache::create();

    let mut render_backend = RenderBackend::new(
        &*window,
        RenderBackendConfig {
            swapchain_extent,
            vsync: window_cfg.vsync,
            graphics_debugging: !opt.no_debug,
        },
    )?;
    let mut render_client = KajiyaRenderClient::new(&render_backend, &lazy_cache)?;
    let mut rg_renderer = kajiya::rg::renderer::Renderer::new(&render_backend)?;

    let mut last_error_text = None;

    let mut camera = kajiya::camera::FirstPersonCamera::new(Vec3::new(0.0, 1.0, 8.0));
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
        let mesh = render_client.world_renderer.add_mesh(mesh);
        render_client
            .world_renderer
            .add_instance(mesh, instance.position.into());
    }

    /*let car_mesh = mmapped_asset::<PackedTriMesh::Flat>("baked/336_lrm.mesh")?;
    let car_mesh = render_client.add_mesh(car_mesh);
    let mut car_pos = Vec3::unit_y() * -0.01;
    let car_inst = render_client.add_instance(car_mesh, car_pos);*/

    render_client
        .world_renderer
        .build_ray_tracing_top_level_acceleration();

    let mut imgui = imgui::Context::create();
    let mut imgui_backend =
        imgui_backend::ImGuiBackend::new(rg_renderer.device().clone(), &window, &mut imgui);
    imgui_backend.create_graphics_resources(swapchain_extent);
    let imgui_backend = Arc::new(Mutex::new(imgui_backend));
    let mut show_gui = true;

    let mut light_theta = -4.54;
    let mut light_phi = 1.48;
    let mut sun_direction_interp = spherical_to_cartesian(light_theta, light_phi);

    const MAX_FPS_LIMIT: u32 = 256;
    let mut max_fps = MAX_FPS_LIMIT;

    let mut last_frame_instant = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        imgui_backend
            .lock()
            .handle_event(window.as_ref(), &mut imgui, &event);

        let ui_wants_mouse = imgui.io().want_capture_mouse;

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
                let dt = dt_duration.as_secs_f32();

                // Limit framerate. Not particularly precise.
                if max_fps != MAX_FPS_LIMIT && dt < 1.0 / max_fps as f32 {
                    std::thread::sleep(std::time::Duration::from_micros(1));
                    return;
                }

                last_frame_instant = now;

                keyboard.update(std::mem::take(&mut keyboard_events), dt);
                mouse_state.update(&new_mouse_state);
                new_mouse_state = mouse_state;

                let input_state = InputState {
                    mouse: mouse_state,
                    keys: keyboard.clone(),
                    dt,
                };
                camera.update(&input_state);

                render_client.world_renderer.store_prev_mesh_transforms();

                // Reset accumulation of the path tracer whenever the camera moves
                if (!camera.is_converged() || keyboard.was_just_pressed(VirtualKeyCode::Back))
                    && render_client.world_renderer.render_mode == RenderMode::Reference
                {
                    render_client.world_renderer.reset_reference_accumulation = true;
                }

                /*if keyboard.is_down(VirtualKeyCode::Z) {
                    car_pos.x += mouse_state.delta.x / 100.0;
                    render_client.set_instance_transform(car_inst, car_pos);
                }*/

                if keyboard.was_just_pressed(VirtualKeyCode::Space) {
                    match render_client.world_renderer.render_mode {
                        RenderMode::Standard => {
                            camera.convergence_sensitivity = 1.0;
                            render_client.world_renderer.render_mode = RenderMode::Reference;
                        }
                        RenderMode::Reference => {
                            camera.convergence_sensitivity = 0.0;
                            render_client.world_renderer.render_mode = RenderMode::Standard;
                        }
                    };
                }

                if !ui_wants_mouse && (mouse_state.button_mask & 1) != 0 {
                    light_theta +=
                        (mouse_state.delta.x / window_cfg.width as f32) * -std::f32::consts::TAU;
                    light_phi +=
                        (mouse_state.delta.y / window_cfg.height as f32) * std::f32::consts::PI;
                }

                //light_phi += dt;
                //light_phi %= std::f32::consts::TAU;

                let sun_direction = spherical_to_cartesian(light_theta, light_phi);
                sun_direction_interp =
                    Vec3::lerp(sun_direction_interp, sun_direction, 0.1).normalize();

                let frame_state = FrameState {
                    camera_matrices: camera.calc_matrices(),
                    window_cfg,
                    //sun_direction: (Vec3::new(-6.0, 4.0, -6.0)).normalize(),
                    sun_direction: sun_direction_interp,
                };

                if keyboard.was_just_pressed(VirtualKeyCode::Tab) {
                    show_gui = !show_gui;
                }

                if keyboard.was_just_pressed(VirtualKeyCode::C) {
                    println!(
                        "position: {}, look_at: {}",
                        frame_state.camera_matrices.eye_position(),
                        frame_state.camera_matrices.eye_position()
                            + frame_state.camera_matrices.eye_direction(),
                    );
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

                        if imgui::CollapsingHeader::new(im_str!("Tweaks"))
                            .default_open(true)
                            .build(&ui)
                        {
                            imgui::Drag::<f32>::new(im_str!("EV shift"))
                                .range(-8.0..=8.0)
                                .speed(0.01)
                                .build(&ui, &mut render_client.world_renderer.ev_shift);
                        }

                        /*if imgui::CollapsingHeader::new(im_str!("csgi"))
                            .default_open(true)
                            .build(&ui)
                        {
                            imgui::Drag::<i32>::new(im_str!("Trace subdivision"))
                                .range(0..=5)
                                .build(&ui, &mut render_client.csgi.trace_subdiv);

                            imgui::Drag::<i32>::new(im_str!("Neighbors per frame"))
                                .range(1..=9)
                                .build(&ui, &mut render_client.csgi.neighbors_per_frame);
                        }*/

                        if imgui::CollapsingHeader::new(im_str!("debug"))
                            .default_open(true)
                            .build(&ui)
                        {
                            if ui.radio_button_bool(
                                im_str!("Scene geometry"),
                                render_client.world_renderer.debug_mode == RenderDebugMode::None,
                            ) {
                                render_client.world_renderer.debug_mode = RenderDebugMode::None;
                            }

                            if ui.radio_button_bool(
                                im_str!("GI voxel grid"),
                                render_client.world_renderer.debug_mode
                                    == RenderDebugMode::CsgiVoxelGrid,
                            ) {
                                render_client.world_renderer.debug_mode =
                                    RenderDebugMode::CsgiVoxelGrid;
                            }

                            imgui::ComboBox::new(im_str!("Shading")).build_simple_string(
                                &ui,
                                &mut render_client.world_renderer.debug_shading_mode,
                                &[
                                    im_str!("Default"),
                                    im_str!("No base color"),
                                    im_str!("Diffuse GI"),
                                    im_str!("Reflections"),
                                    im_str!("RTX OFF"),
                                ],
                            );

                            imgui::Drag::<u32>::new(im_str!("Max FPS"))
                                .range(1..=MAX_FPS_LIMIT)
                                .build(&ui, &mut max_fps);
                        }

                        imgui_backend.prepare_render(&ui, &window);
                        (ui.render(), imgui_backend.get_target_image().unwrap())
                    };

                    let ui_draw_data: &'static imgui::DrawData =
                        unsafe { std::mem::transmute(ui_draw_data) };
                    let imgui_backend = imgui_backend.clone();
                    let gui_extent = swapchain_extent;

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

                match rg_renderer
                    .prepare_frame(|rg| render_client.prepare_render_graph(rg, &frame_state))
                {
                    Ok(()) => {
                        rg_renderer.draw_frame(
                            |dynamic_constants| {
                                render_client
                                    .prepare_frame_constants(dynamic_constants, &frame_state)
                            },
                            &mut render_backend.swapchain,
                        );
                        render_client.retire_frame();
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
