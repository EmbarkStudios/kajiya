mod camera_input;

use anyhow::Context;

use camera_input::InputState;
use imgui::im_str;
use kajiya_simple::{
    cameras::first_person::{CameraController, FirstPersonCamera},
    *,
};

use std::fs::File;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "view", about = "Kajiya scene viewer.")]
struct Opt {
    #[structopt(long, default_value = "1280")]
    width: u32,

    #[structopt(long, default_value = "720")]
    height: u32,

    #[structopt(long, default_value = "1.0")]
    temporal_upsampling: f32,

    #[structopt(long)]
    scene: String,

    #[structopt(long)]
    no_vsync: bool,

    #[structopt(long)]
    no_window_decorations: bool,

    #[structopt(long)]
    no_debug: bool,

    #[structopt(long, default_value = "1.0")]
    gi_volume_scale: f32,
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

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct PersistedAppState {
    camera: FirstPersonCamera,
    light_theta: f32,
    light_phi: f32,
    emissive_multiplier: f32,
    ev_shift: f32,
}

const APP_STATE_CONFIG_FILE_PATH: &str = "view_state.ron";

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();

    let persisted_app_state: Option<PersistedAppState> = File::open(APP_STATE_CONFIG_FILE_PATH)
        .ok()
        .and_then(|f| ron::de::from_reader(f).ok());

    let scene_file = format!("assets/scenes/{}.ron", opt.scene);
    let scene_desc: SceneDesc = ron::de::from_reader(
        File::open(&scene_file).with_context(|| format!("Opening scene file {}", scene_file))?,
    )?;

    let mut kajiya = SimpleMainLoop::builder()
        .resolution([opt.width, opt.height])
        .vsync(!opt.no_vsync)
        .graphics_debugging(!opt.no_debug)
        .temporal_upsampling(opt.temporal_upsampling)
        .default_log_level(log::LevelFilter::Info)
        .build(
            WindowBuilder::new()
                .with_title("kajiya")
                .with_resizable(false)
                .with_decorations(!opt.no_window_decorations),
        )?;

    kajiya.world_renderer.world_gi_scale = opt.gi_volume_scale;

    let mut camera_state = if let Some(persisted_app_state) = &persisted_app_state {
        persisted_app_state.camera.clone()
    } else {
        FirstPersonCamera::new(Vec3::new(0.0, 1.0, 8.0))
    };
    //camera.fov = 65.0;
    //camera.look_smoothness = 20.0;
    //camera.move_smoothness = 20.0;
    camera_state.look_smoothness = 3.0;
    camera_state.move_smoothness = 3.0;

    // Mitsuba match
    /*let mut camera = camera::FirstPersonCamera::new(Vec3::new(-2.0, 4.0, 8.0));
    camera.fov = 35.0 * 9.0 / 16.0;
    camera.look_at(Vec3::new(0.0, 0.75, 0.0));*/

    let lens = CameraLens {
        aspect_ratio: kajiya.window_aspect_ratio(),
        ..Default::default()
    };

    //#[allow(unused_mut)]
    //let mut camera_state = CameraConvergenceEnforcer::new(camera_state);
    //camera_state.convergence_sensitivity = 0.0;
    let camera = &mut camera_state;

    let mut mouse_state: MouseState = Default::default();
    let mut keyboard: KeyboardState = Default::default();

    let mut render_instances = vec![];
    for instance in scene_desc.instances {
        let mesh = kajiya
            .world_renderer
            .add_baked_mesh(format!("/baked/{}.mesh", instance.mesh))?;
        render_instances.push(kajiya.world_renderer.add_instance(
            mesh,
            instance.position.into(),
            Quat::IDENTITY,
        ));
    }

    /*let car_mesh = kajiya
        .world_renderer
        .add_baked_mesh("/baked/336_lrm.mesh")?;
    let mut car_pos = Vec3::Y * -0.01;
    let mut car_rot = 0.0f32;
    let car_inst = kajiya
        .world_renderer
        .add_instance(car_mesh, car_pos, Quat::IDENTITY);*/

    let (
        mut light_theta_state,
        mut light_phi_state,
        mut emissive_multiplier_state,
        mut ev_shift_state,
    ) = if let Some(persisted_app_state) = persisted_app_state.clone() {
        (
            persisted_app_state.light_theta,
            persisted_app_state.light_phi,
            persisted_app_state.emissive_multiplier,
            persisted_app_state.ev_shift,
        )
    } else {
        (-4.54, 1.48, 1.0, 0.0)
    };
    let light_theta = &mut light_theta_state;
    let light_phi = &mut light_phi_state;
    let emissive_multiplier = &mut emissive_multiplier_state;
    let ev_shift = &mut ev_shift_state;

    let mut show_gui = false;
    let mut sun_direction_interp = spherical_to_cartesian(*light_theta, *light_phi);

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
        /*if (!camera.is_converged() || keyboard.was_just_pressed(VirtualKeyCode::Back))
            && ctx.world_renderer.render_mode == RenderMode::Reference
        {
            ctx.world_renderer.reset_reference_accumulation = true;
        }*/

        /*if keyboard.is_down(VirtualKeyCode::Z) {
            car_pos.x += mouse_state.delta.x / 100.0;
        }
        car_rot += 0.5 * ctx.dt;
        ctx.world_renderer.set_instance_transform(
            car_inst,
            car_pos,
            Quat::from_rotation_y(car_rot),
        );*/

        for inst in &render_instances {
            ctx.world_renderer
                .get_instance_dynamic_parameters_mut(*inst)
                .emissive_multiplier = *emissive_multiplier;
        }

        if keyboard.was_just_pressed(VirtualKeyCode::Space) {
            match ctx.world_renderer.render_mode {
                RenderMode::Standard => {
                    //camera.convergence_sensitivity = 1.0;
                    ctx.world_renderer.render_mode = RenderMode::Reference;
                }
                RenderMode::Reference => {
                    //camera.convergence_sensitivity = 0.0;
                    ctx.world_renderer.render_mode = RenderMode::Standard;
                }
            };
        }

        if keyboard.was_just_pressed(VirtualKeyCode::Delete) {
            if let Some(persisted_app_state) = persisted_app_state.as_ref() {
                *light_theta = persisted_app_state.light_theta;
                *light_phi = persisted_app_state.light_phi;
            }
        }

        if mouse_state.button_mask & 1 != 0 {
            *light_theta +=
                (mouse_state.delta.x / ctx.render_extent[0] as f32) * -std::f32::consts::TAU;
            *light_phi +=
                (mouse_state.delta.y / ctx.render_extent[1] as f32) * std::f32::consts::PI;
        }

        //light_phi += dt;
        //light_phi %= std::f32::consts::TAU;

        let sun_direction = spherical_to_cartesian(*light_theta, *light_phi);
        sun_direction_interp = Vec3::lerp(sun_direction_interp, sun_direction, 0.1).normalize();

        let frame_desc = WorldFrameDesc {
            camera_matrices: camera.look().through(&lens),
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
                    for (scope, ms) in gpu_stats.get_ordered() {
                        if scope.name == "debug" {
                            continue;
                        }

                        ui.text(format!("{}: {:.3}ms", scope.name, ms));
                        sum += ms;

                        let hit = ui.is_item_hovered();
                        if hit {
                            ctx.world_renderer.rg_debug_hook = Some(kajiya::rg::GraphDebugHook {
                                render_scope: scope.clone(),
                            });
                        }
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
                        .build(&ui, ev_shift);

                    imgui::Drag::<f32>::new(im_str!("Emissive multiplier"))
                        .range(0.0..=20.0)
                        .speed(0.1)
                        .build(&ui, emissive_multiplier);

                    imgui::Drag::<f32>::new(im_str!("Sun size"))
                        .range(0.0..=10.0)
                        .speed(0.02)
                        .build(&ui, &mut ctx.world_renderer.sun_size_multiplier);
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

        ctx.world_renderer.ev_shift = *ev_shift;

        frame_desc
    })?;

    let app_state = PersistedAppState {
        camera: camera_state.clone(),
        light_theta: light_theta_state,
        light_phi: light_phi_state,
        emissive_multiplier: emissive_multiplier_state,
        ev_shift: ev_shift_state,
    };

    ron::ser::to_writer_pretty(
        File::create(APP_STATE_CONFIG_FILE_PATH)?,
        &app_state,
        Default::default(),
    )?;

    Ok(())
}

fn spherical_to_cartesian(theta: f32, phi: f32) -> Vec3 {
    let x = phi.sin() * theta.cos();
    let y = phi.cos();
    let z = phi.sin() * theta.sin();
    Vec3::new(x, y, z)
}
