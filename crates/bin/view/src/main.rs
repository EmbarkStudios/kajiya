use anyhow::Context;

use dolly::prelude::*;
#[cfg(feature = "use-egui")]
use egui::{CollapsingHeader, ScrollArea};
#[cfg(feature = "dear-imgui")]
use imgui::im_str;
use kajiya::{rg::GraphDebugHook, world_renderer::AddMeshOptions};
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
    //MoveLocalLights,
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
        .fullscreen(opt.fullscreen.then(|| FullscreenMode::Exclusive))
        .build(
            WindowBuilder::new()
                .with_title("kajiya")
                .with_resizable(false)
                .with_decorations(!opt.no_window_decorations),
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
            .with(Position::new(position))
            .with(YawPitch::new().rotation_quat(rotation))
            .with(Smooth::new_position_rotation(1.0, 1.0))
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

    /*let light_mesh = kajiya.world_renderer.add_baked_mesh(
        "/baked/emissive-triangle.mesh",
        AddMeshOptions::new().use_lights(true),
    )?;
    let mut light_instances = Vec::new();*/

    let mut render_instances = vec![];
    for instance in scene_desc.instances {
        let mesh = kajiya.world_renderer.add_baked_mesh(
            format!("/baked/{}.mesh", instance.mesh),
            AddMeshOptions::new(),
        )?;
        render_instances.push(kajiya.world_renderer.add_instance(
            mesh,
            Affine3A::from_rotation_translation(Quat::IDENTITY, instance.position.into()),
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
            camera_position: camera.final_transform.position,
            camera_rotation: camera.final_transform.rotation,
            emissive_multiplier: 1.0,
            vertical_fov: 52.0,
            sun: SunState {
                theta: -4.54,
                phi: 1.48,
            },
            lights: LocalLightsState {
                theta: 1.0,
                phi: 1.0,
                count: 0,
                distance: 1.5,
                multiplier: 10.0,
            },
            ev_shift: 0.0,
        });
    {
        let state = &mut state;

        let mut show_gui = false;
        let mut sun_direction_interp = state.sun.direction();
        let left_click_edit_mode = LeftClickEditMode::MoveSun;

        const MAX_FPS_LIMIT: u32 = 256;
        let mut max_fps = MAX_FPS_LIMIT;
        let mut debug_gi_cascade_idx: u32 = 0;

        let mut locked_rg_debug_hook: Option<GraphDebugHook> = None;

        #[cfg(feature = "use-egui")]
        let mut current_render_scope_name = String::new();

        kajiya.run(move |mut ctx| {
            // Limit framerate. Not particularly precise.
            if max_fps != MAX_FPS_LIMIT {
                std::thread::sleep(std::time::Duration::from_micros(1_000_000 / max_fps as u64));
            }

            keyboard.update(ctx.events);
            mouse.update(ctx.events);

            let input = keymap.map(&keyboard, ctx.dt_filtered);
            let move_vec = camera.final_transform.rotation
                * Vec3::new(input["move_right"], input["move_up"], -input["move_fwd"])
                    .clamp_length_max(1.0)
                * 10.0f32.powf(input["boost"]);

            if (mouse.buttons_held & (1 << 2)) != 0 {
                camera
                    .driver_mut::<YawPitch>()
                    .rotate_yaw_pitch(-0.1 * mouse.delta.x, -0.1 * mouse.delta.y);
            }
            camera
                .driver_mut::<Position>()
                .translate(move_vec * ctx.dt_filtered * 2.5);
            camera.update(ctx.dt_filtered);

            state.camera_position = camera.final_transform.position;
            state.camera_rotation = camera.final_transform.rotation;

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
                        .set_rotation_quat(state.camera_rotation);
                    camera.driver_mut::<Position>().position = state.camera_position;
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
                    } /*LeftClickEditMode::MoveLocalLights => {
                          state.lights.theta += theta_delta;
                          state.lights.phi += phi_delta;
                      }*/
                }
            }

            if keyboard.is_down(VirtualKeyCode::Z) {
                state.lights.distance /= 0.99;
            }
            if keyboard.is_down(VirtualKeyCode::X) {
                state.lights.distance *= 0.99;
            }

            //state.sun.phi += dt;
            //state.sun.phi %= std::f32::consts::TAU;

            let sun_direction = state.sun.direction();
            sun_direction_interp = Vec3::lerp(sun_direction_interp, sun_direction, 0.1).normalize();

            /*#[allow(clippy::comparison_chain)]
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
            }*/

            let lens = CameraLens {
                aspect_ratio: ctx.aspect_ratio(),
                vertical_fov: state.vertical_fov,
                ..Default::default()
            };

            let frame_desc = WorldFrameDesc {
                camera_matrices: camera
                    .final_transform
                    .into_position_rotation()
                    .through(&lens),
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

            ctx.world_renderer.rg_debug_hook = locked_rg_debug_hook.clone();

            #[cfg(feature = "dear-imgui")]
            if show_gui {
                ctx.imgui.take().unwrap().frame(|ui| {
                    if imgui::CollapsingHeader::new(im_str!("Tweaks"))
                        .default_open(true)
                        .build(ui)
                    {
                        imgui::Drag::<f32>::new(im_str!("EV shift"))
                            .range(-8.0..=8.0)
                            .speed(0.01)
                            .build(ui, &mut state.ev_shift);

                        imgui::Drag::<f32>::new(im_str!("Emissive multiplier"))
                            .range(0.0..=10.0)
                            .speed(0.1)
                            .build(ui, &mut state.emissive_multiplier);

                        imgui::Drag::<f32>::new(im_str!("Light intensity multiplier"))
                            .range(0.0..=1000.0)
                            .speed(1.0)
                            .build(ui, &mut state.lights.multiplier);

                        imgui::Drag::<f32>::new(im_str!("Field of view"))
                            .range(1.0..=120.0)
                            .speed(0.25)
                            .build(ui, &mut state.vertical_fov);

                        imgui::Drag::<f32>::new(im_str!("Sun size"))
                            .range(0.0..=10.0)
                            .speed(0.02)
                            .build(ui, &mut ctx.world_renderer.sun_size_multiplier);

                        /*if ui.radio_button_bool(
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
                            .build(ui, &mut state.lights.count);*/

                        #[cfg(feature = "dlss")]
                        {
                            ui.checkbox(im_str!("Use DLSS"), &mut ctx.world_renderer.use_dlss);
                        }
                    }

                    /*if imgui::CollapsingHeader::new(im_str!("csgi"))
                        .default_open(true)
                        .build(ui)
                    {
                        imgui::Drag::<i32>::new(im_str!("Trace subdivision"))
                            .range(0..=5)
                            .build(ui, &mut world_renderer.csgi.trace_subdiv);

                        imgui::Drag::<i32>::new(im_str!("Neighbors per frame"))
                            .range(1..=9)
                            .build(ui, &mut world_renderer.csgi.neighbors_per_frame);
                    }*/

                    if imgui::CollapsingHeader::new(im_str!("Debug"))
                        .default_open(false)
                        .build(ui)
                    {
                        if ui.radio_button_bool(
                            im_str!("Scene geometry"),
                            ctx.world_renderer.debug_mode == RenderDebugMode::None,
                        ) {
                            ctx.world_renderer.debug_mode = RenderDebugMode::None;
                        }

                        if ui.radio_button_bool(
                            im_str!("GI voxel grid"),
                            matches!(
                                ctx.world_renderer.debug_mode,
                                RenderDebugMode::CsgiVoxelGrid { .. }
                            ),
                        ) {
                            ctx.world_renderer.debug_mode = RenderDebugMode::CsgiVoxelGrid {
                                cascade_idx: debug_gi_cascade_idx as _,
                            };
                        }

                        if matches!(
                            ctx.world_renderer.debug_mode,
                            RenderDebugMode::CsgiVoxelGrid { .. }
                        ) {
                            imgui::Drag::<u32>::new(im_str!("Cascade index"))
                                .range(0..=3)
                                .build(ui, &mut debug_gi_cascade_idx);

                            ctx.world_renderer.debug_mode = RenderDebugMode::CsgiVoxelGrid {
                                cascade_idx: debug_gi_cascade_idx as _,
                            };
                        }

                        if ui.radio_button_bool(
                            im_str!("GI voxel radiance"),
                            ctx.world_renderer.debug_mode == RenderDebugMode::CsgiRadiance,
                        ) {
                            ctx.world_renderer.debug_mode = RenderDebugMode::CsgiRadiance;
                        }

                        imgui::ComboBox::new(im_str!("Shading")).build_simple_string(
                            ui,
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
                            .build(ui, &mut max_fps);
                    }

                    if imgui::CollapsingHeader::new(im_str!("GPU passes"))
                        .default_open(true)
                        .build(ui)
                    {
                        let gpu_stats = gpu_profiler::get_stats();
                        ui.text(format!("CPU frame time: {:.3}ms", ctx.dt_filtered * 1000.0));

                        let ordered_scopes = gpu_stats.get_ordered();
                        let gpu_time_ms: f64 = ordered_scopes.iter().map(|(_, ms)| ms).sum();

                        ui.text(format!("GPU frame time: {:.3}ms", gpu_time_ms));

                        for (scope, ms) in ordered_scopes {
                            if scope.name == "debug" || scope.name.starts_with('_') {
                                continue;
                            }

                            let style = locked_rg_debug_hook.as_ref().and_then(|hook| {
                                if hook.render_scope == scope {
                                    Some(ui.push_style_color(
                                        imgui::StyleColor::Text,
                                        [1.0, 1.0, 0.1, 1.0],
                                    ))
                                } else {
                                    None
                                }
                            });

                            ui.text(format!("{}: {:.3}ms", scope.name, ms));

                            if let Some(style) = style {
                                style.pop(ui);
                            }

                            if ui.is_item_hovered() {
                                ctx.world_renderer.rg_debug_hook =
                                    Some(kajiya::rg::GraphDebugHook {
                                        render_scope: scope.clone(),
                                    });

                                if ui.is_item_clicked(imgui::MouseButton::Left) {
                                    if locked_rg_debug_hook == ctx.world_renderer.rg_debug_hook {
                                        locked_rg_debug_hook = None;
                                    } else {
                                        locked_rg_debug_hook =
                                            ctx.world_renderer.rg_debug_hook.clone();
                                    }
                                }
                            }
                        }
                    }
                });
            }

            #[cfg(feature = "use-egui")]
            if show_gui {
                let egui = ctx.egui.as_mut().unwrap();
                egui.frame(&mouse, |egui_ctx| {
                    egui::Window::new("Debug")
                        .resizable(true)
                        .min_height(500.0)
                        .show(egui_ctx, |ui| {
                            ScrollArea::vertical()
                                .auto_shrink([false; 2])
                                .show(ui, |ui| {
                                    CollapsingHeader::new("Tweaks").default_open(true).show(
                                        ui,
                                        |ui| {
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::Slider::new(
                                                        &mut state.ev_shift,
                                                        -8.0..=8.0,
                                                    )
                                                    .clamp_to_range(false)
                                                    .smart_aim(false)
                                                    .step_by(0.01),
                                                );
                                                ui.label("EV shift");
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::Slider::new(
                                                        &mut state.emissive_multiplier,
                                                        0.0..=10.0,
                                                    )
                                                    .clamp_to_range(false)
                                                    .smart_aim(false)
                                                    .step_by(0.01),
                                                );
                                                ui.label("Emissive multiplier");
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::Slider::new(
                                                        &mut state.lights.multiplier,
                                                        0.0..=1000.0,
                                                    )
                                                    .clamp_to_range(false)
                                                    .smart_aim(false)
                                                    .step_by(0.01),
                                                );
                                                ui.label("Light intensity multiplier");
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::Slider::new(
                                                        &mut state.vertical_fov,
                                                        0.0..=120.0,
                                                    )
                                                    .clamp_to_range(false)
                                                    .smart_aim(false)
                                                    .step_by(0.01),
                                                );
                                                ui.label("Field of view");
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::Slider::new(
                                                        &mut ctx.world_renderer.sun_size_multiplier,
                                                        0.0..=10.0,
                                                    )
                                                    .clamp_to_range(false)
                                                    .smart_aim(false)
                                                    .step_by(0.01),
                                                );
                                                ui.label("Sun size");
                                            });
                                        },
                                    );

                                    CollapsingHeader::new("Debug").default_open(false).show(
                                        ui,
                                        |ui| {
                                            ui.radio_value(
                                                &mut ctx.world_renderer.debug_mode,
                                                RenderDebugMode::None,
                                                "Scene geometry",
                                            );
                                            ui.radio_value(
                                                &mut ctx.world_renderer.debug_mode,
                                                RenderDebugMode::CsgiVoxelGrid {
                                                    cascade_idx: debug_gi_cascade_idx as usize,
                                                },
                                                "GI voxel grid",
                                            );
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::DragValue::new(&mut debug_gi_cascade_idx)
                                                        .clamp_range(0..=3),
                                                );
                                                if debug_gi_cascade_idx > 0 {
                                                    ctx.world_renderer.debug_mode =
                                                        RenderDebugMode::CsgiVoxelGrid {
                                                            cascade_idx: debug_gi_cascade_idx
                                                                as usize,
                                                        };
                                                }
                                                ui.label("Cascade index");
                                            });
                                            ui.radio_value(
                                                &mut ctx.world_renderer.debug_mode,
                                                RenderDebugMode::CsgiRadiance,
                                                "GI voxel radiance",
                                            );

                                            egui::ComboBox::from_label("Shading")
                                                .selected_text(format!(
                                                    "{}",
                                                    match ctx.world_renderer.debug_shading_mode {
                                                        0 => "Default",
                                                        1 => "No base color",
                                                        2 => "Diffuse GI",
                                                        3 => "Reflections",
                                                        4 => "RTX OFF",
                                                        _ => "None",
                                                    }
                                                ))
                                                .show_ui(ui, |ui| {
                                                    ui.selectable_value(
                                                        &mut ctx.world_renderer.debug_shading_mode,
                                                        0,
                                                        "Default",
                                                    );
                                                    ui.selectable_value(
                                                        &mut ctx.world_renderer.debug_shading_mode,
                                                        1,
                                                        "No base color",
                                                    );
                                                    ui.selectable_value(
                                                        &mut ctx.world_renderer.debug_shading_mode,
                                                        2,
                                                        "Diffuse GI",
                                                    );
                                                    ui.selectable_value(
                                                        &mut ctx.world_renderer.debug_shading_mode,
                                                        3,
                                                        "Reflections",
                                                    );
                                                    ui.selectable_value(
                                                        &mut ctx.world_renderer.debug_shading_mode,
                                                        4,
                                                        "RTX OFF",
                                                    );
                                                });

                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::DragValue::new(&mut max_fps)
                                                        .clamp_range(1..=MAX_FPS_LIMIT),
                                                );
                                                ui.label("Max FPS");
                                            });
                                        },
                                    );

                                    CollapsingHeader::new("GPU Passes").default_open(true).show(
                                        ui,
                                        |ui| {
                                            let gpu_stats = gpu_profiler::get_stats();
                                            ui.label(format!(
                                                "CPU frame time: {:.3}ms",
                                                ctx.dt_filtered * 1000.0
                                            ));

                                            let ordered_scopes = gpu_stats.get_ordered();
                                            let gpu_time_ms: f64 =
                                                ordered_scopes.iter().map(|(_, ms)| ms).sum();

                                            ui.label(format!(
                                                "GPU frame time: {:.3}ms",
                                                gpu_time_ms
                                            ));
                                            for (scope, ms) in ordered_scopes {
                                                if scope.name == "debug"
                                                    || scope.name.starts_with('_')
                                                {
                                                    continue;
                                                }

                                                let label = ui.selectable_label(
                                                    scope.name == current_render_scope_name,
                                                    format!("{}: {:.3}ms", scope.name, ms),
                                                );

                                                if label.hovered() {
                                                    ctx.world_renderer.rg_debug_hook =
                                                        Some(kajiya::rg::GraphDebugHook {
                                                            render_scope: scope.clone(),
                                                        });

                                                    if label.clicked() {
                                                        if current_render_scope_name == scope.name {
                                                            current_render_scope_name.clear();
                                                        } else {
                                                            current_render_scope_name = scope.name;
                                                        }
                                                        if locked_rg_debug_hook
                                                            == ctx.world_renderer.rg_debug_hook
                                                        {
                                                            locked_rg_debug_hook = None;
                                                        } else {
                                                            locked_rg_debug_hook = ctx
                                                                .world_renderer
                                                                .rg_debug_hook
                                                                .clone();
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                    );
                                });
                        });
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
