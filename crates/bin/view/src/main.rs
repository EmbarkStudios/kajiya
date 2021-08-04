use anyhow::Context;

use dolly::prelude::*;
use imgui::im_str;
use kajiya::world_renderer::AddMeshOptions;
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

    #[structopt(long, default_value = "1.0")]
    temporal_upsampling: f32,

    #[structopt(long)]
    scene: String,

    #[structopt(long)]
    no_vsync: bool,

    #[structopt(long)]
    no_window_decorations: bool,

    #[structopt(long)]
    fullscreen: bool,

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
struct SunState {
    theta: f32,
    phi: f32,
}

impl SunState {
    pub fn direction(&self) -> Vec3 {
        fn spherical_to_cartesian(theta: f32, phi: f32) -> Vec3 {
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            Vec3::new(x, y, z)
        }

        spherical_to_cartesian(self.theta, self.phi)
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct LocalLightsState {
    theta: f32,
    phi: f32,
    count: u32,
    distance: f32,
    multiplier: f32,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct PersistedAppState {
    camera_position: Vec3,
    camera_rotation: Quat,
    vertical_fov: f32,
    emissive_multiplier: f32,
    sun: SunState,
    lights: LocalLightsState,
    ev_shift: f32,
}

#[derive(PartialEq, Eq)]
enum LeftClickEditMode {
    MoveSun,
    MoveLocalLights,
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
                .with_decorations(!opt.no_window_decorations)
                .with_fullscreen(
                    opt.fullscreen
                        .then(|| winit::window::Fullscreen::Borderless(None)),
                ),
        )?;

    kajiya.world_renderer.world_gi_scale = opt.gi_volume_scale;

    // Mitsuba match
    /*let mut camera = camera::FirstPersonCamera::new(Vec3::new(-2.0, 4.0, 8.0));
    camera.fov = 35.0 * 9.0 / 16.0;
    camera.look_at(Vec3::new(0.0, 0.75, 0.0));*/

    //#[allow(unused_mut)]
    //let mut camera_state = CameraConvergenceEnforcer::new(camera_state);
    //camera_state.convergence_sensitivity = 0.0;
    //let camera = &mut camera_state;

    let mut camera = {
        let (position, rotation) = if let Some(state) = &persisted_app_state {
            (state.camera_position, state.camera_rotation)
        } else {
            (Vec3::new(0.0, 1.0, 8.0), Quat::IDENTITY)
        };

        CameraRig::builder()
            .with(Positional::new(position))
            .with(YawPitch::new().rotation(rotation))
            .with(Smooth::new_move_look(1.0, 1.0))
            .build()
    };

    let mut mouse: MouseState = Default::default();
    let mut keyboard: KeyboardState = Default::default();

    let mut keymap = KeyboardMap::new()
        .bind(VirtualKeyCode::W, KeyMap::new("move_fwd", 1.0))
        .bind(VirtualKeyCode::S, KeyMap::new("move_fwd", -1.0))
        .bind(VirtualKeyCode::A, KeyMap::new("move_right", -1.0))
        .bind(VirtualKeyCode::D, KeyMap::new("move_right", 1.0))
        .bind(VirtualKeyCode::Q, KeyMap::new("move_up", -1.0))
        .bind(VirtualKeyCode::E, KeyMap::new("move_up", 1.0))
        .bind(VirtualKeyCode::LShift, KeyMap::new("boost", 1.0))
        .bind(VirtualKeyCode::LControl, KeyMap::new("boost", -1.0));

    let light_mesh = kajiya.world_renderer.add_baked_mesh(
        "/baked/emissive-triangle.mesh",
        AddMeshOptions::new().use_lights(true),
    )?;
    let mut light_instances = Vec::new();

    let mut render_instances = vec![];
    for instance in scene_desc.instances {
        let mesh = kajiya.world_renderer.add_baked_mesh(
            format!("/baked/{}.mesh", instance.mesh),
            AddMeshOptions::new(),
        )?;
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

    let mut state = persisted_app_state
        .clone()
        .unwrap_or_else(|| PersistedAppState {
            camera_position: camera.transform.translation,
            camera_rotation: camera.transform.rotation,
            emissive_multiplier: 1.0,
            vertical_fov: 52.0,
            sun: SunState {
                theta: -4.54,
                phi: 1.48,
            },
            lights: LocalLightsState {
                theta: 1.0,
                phi: 1.0,
                count: 1,
                distance: 1.5,
                multiplier: 10.0,
            },
            ev_shift: 0.0,
        });
    {
        let state = &mut state;

        let mut show_gui = false;
        let mut sun_direction_interp = state.sun.direction();
        let mut left_click_edit_mode = LeftClickEditMode::MoveSun;

        const MAX_FPS_LIMIT: u32 = 256;
        let mut max_fps = MAX_FPS_LIMIT;

        kajiya.run(move |mut ctx| {
            // Limit framerate. Not particularly precise.
            if max_fps != MAX_FPS_LIMIT {
                std::thread::sleep(std::time::Duration::from_micros(1_000_000 / max_fps as u64));
            }

            keyboard.update(&ctx.events);
            mouse.update(&ctx.events);

            let input = keymap.map(&keyboard, ctx.dt);
            let move_vec = camera.transform.rotation
                * Vec3::new(input["move_right"], input["move_up"], -input["move_fwd"])
                    .clamp_length_max(1.0)
                * 10.0f32.powf(input["boost"]);

            if (mouse.buttons_held & (1 << 2)) != 0 {
                camera
                    .driver_mut::<YawPitch>()
                    .rotate_yaw_pitch(-0.1 * mouse.delta.x, -0.1 * mouse.delta.y);
            }
            camera
                .driver_mut::<Positional>()
                .translate(move_vec * ctx.dt * 2.5);
            camera.update(ctx.dt);

            state.camera_position = camera.transform.translation;
            state.camera_rotation = camera.transform.rotation;

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
                    .emissive_multiplier = state.emissive_multiplier;
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
                    *state = persisted_app_state.clone();

                    camera
                        .driver_mut::<YawPitch>()
                        .set_rotation(state.camera_rotation);
                    camera.driver_mut::<Positional>().position = state.camera_position;
                }
            }

            if mouse.buttons_held & 1 != 0 {
                let theta_delta =
                    (mouse.delta.x / ctx.render_extent[0] as f32) * -std::f32::consts::TAU;
                let phi_delta =
                    (mouse.delta.y / ctx.render_extent[1] as f32) * std::f32::consts::PI;

                match left_click_edit_mode {
                    LeftClickEditMode::MoveSun => {
                        state.sun.theta += theta_delta;
                        state.sun.phi += phi_delta;
                    }
                    LeftClickEditMode::MoveLocalLights => {
                        state.lights.theta += theta_delta;
                        state.lights.phi += phi_delta;
                    }
                }
            }

            if keyboard.is_down(VirtualKeyCode::Z) {
                state.lights.distance /= 0.95;
            }
            if keyboard.is_down(VirtualKeyCode::X) {
                state.lights.distance *= 0.95;
            }

            //state.sun.phi += dt;
            //state.sun.phi %= std::f32::consts::TAU;

            let sun_direction = state.sun.direction();
            sun_direction_interp = Vec3::lerp(sun_direction_interp, sun_direction, 0.1).normalize();

            if light_instances.len() > state.lights.count as usize {
                for extra_light in light_instances.drain(state.lights.count as usize..) {
                    ctx.world_renderer.remove_instance(extra_light);
                }
            } else if light_instances.len() < state.lights.count as usize {
                light_instances.extend(
                    (0..(state.lights.count as usize - light_instances.len())).map(|_| {
                        ctx.world_renderer
                            .add_instance(light_mesh, Vec3::ZERO, Quat::IDENTITY)
                    }),
                );
            }

            for (i, inst) in light_instances.iter().enumerate() {
                let ring_rot = Quat::from_rotation_y(
                    (i as f32) / light_instances.len() as f32 * std::f32::consts::TAU,
                );

                let rot =
                    Quat::from_euler(EulerRot::YXZ, -state.lights.theta, -state.lights.phi, 0.0)
                        * ring_rot;
                ctx.world_renderer.set_instance_transform(
                    *inst,
                    rot * (Vec3::Z * state.lights.distance) + Vec3::new(0.1, 1.2, 0.0),
                    rot,
                );

                ctx.world_renderer
                    .get_instance_dynamic_parameters_mut(*inst)
                    .emissive_multiplier = state.lights.multiplier;
            }

            let lens = CameraLens {
                aspect_ratio: ctx.aspect_ratio(),
                vertical_fov: state.vertical_fov,
                ..Default::default()
            };

            let frame_desc = WorldFrameDesc {
                camera_matrices: camera.transform.into_translation_rotation().through(&lens),
                render_extent: ctx.render_extent,
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
                    if imgui::CollapsingHeader::new(im_str!("Tweaks"))
                        .default_open(true)
                        .build(&ui)
                    {
                        imgui::Drag::<f32>::new(im_str!("EV shift"))
                            .range(-8.0..=8.0)
                            .speed(0.01)
                            .build(&ui, &mut state.ev_shift);

                        imgui::Drag::<f32>::new(im_str!("Emissive multiplier"))
                            .range(0.0..=10.0)
                            .speed(0.1)
                            .build(&ui, &mut state.emissive_multiplier);

                        imgui::Drag::<f32>::new(im_str!("Light intensity multiplier"))
                            .range(0.0..=1000.0)
                            .speed(1.0)
                            .build(&ui, &mut state.lights.multiplier);

                        imgui::Drag::<f32>::new(im_str!("Field of view"))
                            .range(1.0..=120.0)
                            .speed(0.25)
                            .build(&ui, &mut state.vertical_fov);

                        imgui::Drag::<f32>::new(im_str!("Sun size"))
                            .range(0.0..=10.0)
                            .speed(0.02)
                            .build(&ui, &mut ctx.world_renderer.sun_size_multiplier);

                        ui.checkbox(
                            im_str!("Reflection reservoir resampling"),
                            &mut ctx.world_renderer.rtr.reservoir_resampling,
                        );

                        if ui.radio_button_bool(
                            im_str!("Move sun"),
                            left_click_edit_mode == LeftClickEditMode::MoveSun,
                        ) {
                            left_click_edit_mode = LeftClickEditMode::MoveSun;
                        }

                        if ui.radio_button_bool(
                            im_str!("Move local lights"),
                            left_click_edit_mode == LeftClickEditMode::MoveLocalLights,
                        ) {
                            left_click_edit_mode = LeftClickEditMode::MoveLocalLights;
                        }

                        imgui::Drag::<u32>::new(im_str!("Light count"))
                            .range(0..=10)
                            .build(&ui, &mut state.lights.count);

                        #[cfg(feature = "dlss")]
                        {
                            ui.checkbox(im_str!("Use DLSS"), &mut ctx.world_renderer.use_dlss);
                        }
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

                    if imgui::CollapsingHeader::new(im_str!("Debug"))
                        .default_open(false)
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

                        if ui.radio_button_bool(
                            im_str!("GI voxel radiance"),
                            ctx.world_renderer.debug_mode == RenderDebugMode::CsgiRadiance,
                        ) {
                            ctx.world_renderer.debug_mode = RenderDebugMode::CsgiRadiance;
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

                    if imgui::CollapsingHeader::new(im_str!("GPU passes"))
                        .default_open(true)
                        .build(ui)
                    {
                        let gpu_stats = gpu_profiler::get_stats();
                        ui.text(format!("CPU frame time: {:.3}ms", ctx.dt * 1000.0));

                        let ordered_scopes = gpu_stats.get_ordered();
                        let gpu_time_ms: f64 = ordered_scopes.iter().map(|(_, ms)| ms).sum();

                        ui.text(format!("GPU frame time: {:.3}ms", gpu_time_ms));

                        for (scope, ms) in ordered_scopes {
                            if scope.name == "debug" || scope.name.starts_with("_") {
                                continue;
                            }

                            ui.text(format!("{}: {:.3}ms", scope.name, ms));

                            let hit = ui.is_item_hovered();
                            if hit {
                                ctx.world_renderer.rg_debug_hook =
                                    Some(kajiya::rg::GraphDebugHook {
                                        render_scope: scope.clone(),
                                    });
                            }
                        }
                    }
                });
            }

            ctx.world_renderer.ev_shift = state.ev_shift;

            frame_desc
        })?;
    }

    ron::ser::to_writer_pretty(
        File::create(APP_STATE_CONFIG_FILE_PATH)?,
        &state,
        Default::default(),
    )?;

    Ok(())
}
