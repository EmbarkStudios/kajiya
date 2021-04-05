mod camera_input;

use anyhow::Context;

use camera_input::InputState;
use imgui::im_str;
use kajiya_simple::*;

use std::fs::File;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "view", about = "Kajiya scene viewer.")]
struct Opt {
    #[structopt(long, default_value = "1280")]
    width: u32,

    #[structopt(long, default_value = "720")]
    height: u32,

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
    let opt = Opt::from_args();

    let scene_file = format!("assets/scenes/{}.ron", opt.scene);
    let scene_desc: SceneDesc = ron::de::from_reader(
        File::open(&scene_file).with_context(|| format!("Opening scene file {}", scene_file))?,
    )?;

    let mut kajiya = SimpleMainLoop::builder()
        .vsync(!opt.no_vsync)
        .graphics_debugging(!opt.no_debug)
        .default_log_level(log::LevelFilter::Info)
        .build(
            WindowBuilder::new()
                .with_title("kajiya")
                .with_resizable(false)
                .with_decorations(!opt.no_window_decorations)
                .with_inner_size(winit::dpi::LogicalSize::new(
                    opt.width as f64,
                    opt.height as f64,
                )),
        )?;

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

    camera.aspect = kajiya.window_aspect_ratio();

    #[allow(unused_mut)]
    let mut camera = CameraConvergenceEnforcer::new(camera);
    camera.convergence_sensitivity = 0.0;

    let mut mouse_state: MouseState = Default::default();
    let mut keyboard: KeyboardState = Default::default();

    for instance in scene_desc.instances {
        let mesh = kajiya
            .world_renderer
            .add_baked_mesh(format!("/baked/{}.mesh", instance.mesh))?;
        kajiya
            .world_renderer
            .add_instance(mesh, instance.position.into(), Quat::identity());
    }

    /*let car_mesh = kajiya
        .world_renderer
        .add_baked_mesh("/baked/336_lrm.mesh")?;
    let mut car_pos = Vec3::unit_y() * -0.01;
    let mut car_rot = 0.0f32;
    let car_inst = kajiya
        .world_renderer
        .add_instance(car_mesh, car_pos, Quat::identity());*/

    let mut show_gui = true;
    let mut light_theta = -4.54;
    let mut light_phi = 1.48;
    let mut sun_direction_interp = spherical_to_cartesian(light_theta, light_phi);

    const MAX_FPS_LIMIT: u32 = 256;
    let mut max_fps = MAX_FPS_LIMIT;

    let mut new_mouse_state: MouseState = Default::default();

    kajiya.run(move |mut ctx| {
        // TODO
        /*// Limit framerate. Not particularly precise.
        if max_fps != MAX_FPS_LIMIT && dt < 1.0 / max_fps as f32 {
            std::thread::sleep(std::time::Duration::from_micros(1));
            return;
        }*/

        let mut keyboard_events: Vec<KeyboardInput> = Vec::new();

        for event in ctx.events {
            match event {
                WindowEvent::KeyboardInput { input, .. } => {
                    keyboard_events.push(*input);
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
            }
        }

        keyboard.update(std::mem::take(&mut keyboard_events), ctx.dt);
        mouse_state.update(&new_mouse_state);
        new_mouse_state = mouse_state;

        let input_state = InputState {
            mouse: mouse_state,
            keys: keyboard.clone(),
            dt: ctx.dt,
        };
        camera.update(&input_state);

        // Reset accumulation of the path tracer whenever the camera moves
        if (!camera.is_converged() || keyboard.was_just_pressed(VirtualKeyCode::Back))
            && ctx.world_renderer.render_mode == RenderMode::Reference
        {
            ctx.world_renderer.reset_reference_accumulation = true;
        }

        /*if keyboard.is_down(VirtualKeyCode::Z) {
            car_pos.x += mouse_state.delta.x / 100.0;
        }
        car_rot += 0.5 * ctx.dt;
        ctx.world_renderer.set_instance_transform(
            car_inst,
            car_pos,
            Quat::from_rotation_y(car_rot),
        );*/

        if keyboard.was_just_pressed(VirtualKeyCode::Space) {
            match ctx.world_renderer.render_mode {
                RenderMode::Standard => {
                    camera.convergence_sensitivity = 1.0;
                    ctx.world_renderer.render_mode = RenderMode::Reference;
                }
                RenderMode::Reference => {
                    camera.convergence_sensitivity = 0.0;
                    ctx.world_renderer.render_mode = RenderMode::Standard;
                }
            };
        }

        if mouse_state.button_mask & 1 != 0 {
            light_theta +=
                (mouse_state.delta.x / ctx.render_extent[0] as f32) * -std::f32::consts::TAU;
            light_phi += (mouse_state.delta.y / ctx.render_extent[1] as f32) * std::f32::consts::PI;
        }

        //light_phi += dt;
        //light_phi %= std::f32::consts::TAU;

        let sun_direction = spherical_to_cartesian(light_theta, light_phi);
        sun_direction_interp = Vec3::lerp(sun_direction_interp, sun_direction, 0.1).normalize();

        let frame_desc = WorldFrameDesc {
            camera_matrices: camera.calc_matrices(),
            render_extent: ctx.render_extent,
            //sun_direction: (Vec3::new(-6.0, 4.0, -6.0)).normalize(),
            sun_direction: sun_direction_interp,
        };

        if keyboard.was_just_pressed(VirtualKeyCode::Tab) {
            show_gui = !show_gui;
        }

        if keyboard.was_just_pressed(VirtualKeyCode::C) {
            println!(
                "position: {}, look_at: {}",
                frame_desc.camera_matrices.eye_position(),
                frame_desc.camera_matrices.eye_position()
                    + frame_desc.camera_matrices.eye_direction(),
            );
        }

        if show_gui {
            ctx.imgui.take().unwrap().frame(|ui| {
                if imgui::CollapsingHeader::new(im_str!("GPU passes"))
                    .default_open(true)
                    .build(ui)
                {
                    let gpu_stats = gpu_profiler::get_stats();
                    ui.text(format!("CPU frame time: {:.3}ms", ctx.dt * 1000.0));

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
                        .build(&ui, &mut ctx.world_renderer.ev_shift);
                }

                /*if imgui::CollapsingHeader::new(im_str!("csgi"))
                    .default_open(true)
                    .build(&ui)
                {
                    imgui::Drag::<i32>::new(im_str!("Trace subdivision"))
                        .range(0..=5)
                        .build(&ui, &mut world_renderer.csgi.trace_subdiv);

                    imgui::Drag::<i32>::new(im_str!("Neighbors per frame"))
                        .range(1..=9)
                        .build(&ui, &mut world_renderer.csgi.neighbors_per_frame);
                }*/

                if imgui::CollapsingHeader::new(im_str!("debug"))
                    .default_open(true)
                    .build(&ui)
                {
                    if ui.radio_button_bool(
                        im_str!("Scene geometry"),
                        ctx.world_renderer.debug_mode == RenderDebugMode::None,
                    ) {
                        ctx.world_renderer.debug_mode = RenderDebugMode::None;
                    }

                    if ui.radio_button_bool(
                        im_str!("GI voxel grid"),
                        ctx.world_renderer.debug_mode == RenderDebugMode::CsgiVoxelGrid,
                    ) {
                        ctx.world_renderer.debug_mode = RenderDebugMode::CsgiVoxelGrid;
                    }

                    imgui::ComboBox::new(im_str!("Shading")).build_simple_string(
                        &ui,
                        &mut ctx.world_renderer.debug_shading_mode,
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
            });
        }

        frame_desc
    })
}

fn spherical_to_cartesian(theta: f32, phi: f32) -> Vec3 {
    let x = phi.sin() * theta.cos();
    let y = phi.cos();
    let z = phi.sin() * theta.sin();
    Vec3::new(x, y, z)
}
