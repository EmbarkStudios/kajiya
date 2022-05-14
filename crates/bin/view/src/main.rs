use anyhow::Context;

use dolly::prelude::*;
use imgui::im_str;
use kajiya::{
    rg::GraphDebugHook,
    world_renderer::{AddMeshOptions, MeshHandle},
};
use kajiya_simple::*;

use std::{collections::HashMap, fs::File, path::PathBuf};
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

    #[structopt(long)]
    physical_device_index: Option<usize>,

    #[structopt(long, default_value = "1.0")]
    gi_volume_scale: f32,
}

#[derive(serde::Deserialize)]
struct SceneDesc {
    instances: Vec<SceneInstanceDesc>,
}

fn default_instance_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

#[derive(serde::Deserialize)]
struct SceneInstanceDesc {
    position: [f32; 3],
    #[serde(default = "default_instance_scale")]
    scale: [f32; 3],
    mesh: String,
}

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct LocalLightsState {
    theta: f32,
    phi: f32,
    count: u32,
    distance: f32,
    multiplier: f32,
}

fn always_true() -> bool {
    true
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct PersistedAppState {
    camera_position: Vec3,
    camera_rotation: Quat,
    vertical_fov: f32,
    emissive_multiplier: f32,
    #[serde(default = "always_true")]
    enable_emissive: bool,
    sun: SunState,
    lights: LocalLightsState,
    ev_shift: f32,
    #[serde(default)]
    use_dynamic_exposure: bool,
    #[serde(default = "default_camera_speed")]
    camera_speed: f32,
    #[serde(default = "default_camera_smoothness")]
    camera_smoothness: f32,
    #[serde(default = "default_sun_rotation_smoothness")]
    sun_rotation_smoothness: f32,
}

impl PersistedAppState {
    fn should_reset_path_tracer(&self, other: &PersistedAppState) -> bool {
        false
            || !self
                .camera_position
                .abs_diff_eq(other.camera_position, 1e-5)
            || !self
                .camera_rotation
                .abs_diff_eq(other.camera_rotation, 1e-5)
            || self.vertical_fov != other.vertical_fov
            || self.emissive_multiplier != other.emissive_multiplier
            || self.sun != other.sun
            || self.lights != other.lights
    }
}

fn default_camera_speed() -> f32 {
    2.5
}

fn default_camera_smoothness() -> f32 {
    1.0
}

fn default_sun_rotation_smoothness() -> f32 {
    1.0
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
        .physical_device_index(opt.physical_device_index)
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

    let mut camera = {
        let (position, rotation) = if let Some(state) = &persisted_app_state {
            (state.camera_position, state.camera_rotation)
        } else {
            (Vec3::new(0.0, 1.0, 8.0), Quat::IDENTITY)
        };

        CameraRig::builder()
            .with(Position::new(position))
            .with(YawPitch::new().rotation_quat(rotation))
            .with(Smooth::default())
            .build()
    };

    let mut mouse: MouseState = Default::default();
    let mut keyboard: KeyboardState = Default::default();
    let mut reset_path_tracer = false;

    let mut keymap = KeyboardMap::new()
        .bind(VirtualKeyCode::W, KeyMap::new("move_fwd", 1.0))
        .bind(VirtualKeyCode::S, KeyMap::new("move_fwd", -1.0))
        .bind(VirtualKeyCode::A, KeyMap::new("move_right", -1.0))
        .bind(VirtualKeyCode::D, KeyMap::new("move_right", 1.0))
        .bind(VirtualKeyCode::Q, KeyMap::new("move_up", -1.0))
        .bind(VirtualKeyCode::E, KeyMap::new("move_up", 1.0))
        .bind(
            VirtualKeyCode::LShift,
            KeyMap::new("boost", 1.0).activation_time(0.25),
        )
        .bind(
            VirtualKeyCode::LControl,
            KeyMap::new("boost", -1.0).activation_time(0.5),
        );

    /*let light_mesh = kajiya.world_renderer.add_baked_mesh(
        "/baked/emissive-triangle.mesh",
        AddMeshOptions::new().use_lights(true),
    )?;
    let mut light_instances = Vec::new();*/

    let mut known_meshes: HashMap<PathBuf, MeshHandle> = HashMap::new();

    let mut render_instances = vec![];
    for instance in scene_desc.instances {
        let path = PathBuf::from(format!("/baked/{}.mesh", instance.mesh));

        let mesh = *known_meshes.entry(path.clone()).or_insert_with(|| {
            kajiya
                .world_renderer
                .add_baked_mesh(path, AddMeshOptions::new())
                .unwrap()
        });

        render_instances.push(kajiya.world_renderer.add_instance(
            mesh,
            Affine3A::from_scale_rotation_translation(
                instance.scale.into(),
                Quat::IDENTITY,
                instance.position.into(),
            ),
        ));
    }

    let car_mesh = kajiya
        .world_renderer
        .add_baked_mesh("/baked/336_lrm.mesh", AddMeshOptions::default())?;
    let mut car_pos = Vec3::Y * -0.01;
    let car_rot = 0.0f32;
    let car_inst = kajiya.world_renderer.add_instance(
        car_mesh,
        Affine3A::from_rotation_translation(Quat::IDENTITY, car_pos),
    );

    let mut state = persisted_app_state
        .clone()
        .unwrap_or_else(|| PersistedAppState {
            camera_position: camera.final_transform.position,
            camera_rotation: camera.final_transform.rotation,
            emissive_multiplier: 1.0,
            enable_emissive: true,
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
            use_dynamic_exposure: false,
            camera_speed: default_camera_speed(),
            camera_smoothness: default_camera_smoothness(),
            sun_rotation_smoothness: default_sun_rotation_smoothness(),
        });
    {
        let state = &mut state;

        let mut show_gui = false;
        let mut sun_direction_interp = state.sun.direction();
        let left_click_edit_mode = LeftClickEditMode::MoveSun;

        const MAX_FPS_LIMIT: u32 = 256;
        let mut max_fps = MAX_FPS_LIMIT;

        let mut locked_rg_debug_hook: Option<GraphDebugHook> = None;
        let mut grab_cursor_pos = winit::dpi::PhysicalPosition::default();

        kajiya.run(move |mut ctx| {
            // Limit framerate. Not particularly precise.
            if max_fps != MAX_FPS_LIMIT {
                std::thread::sleep(std::time::Duration::from_micros(1_000_000 / max_fps as u64));
            }

            let prev_state = state.clone();

            let smooth = camera.driver_mut::<Smooth>();
            if ctx.world_renderer.render_mode == RenderMode::Reference {
                smooth.position_smoothness = 0.0;
                smooth.rotation_smoothness = 0.0;
            } else {
                smooth.position_smoothness = state.camera_smoothness;
                smooth.rotation_smoothness = state.camera_smoothness;
            }

            keyboard.update(ctx.events);
            mouse.update(ctx.events);

            // When starting camera rotation, hide the mouse cursor, and capture it to the window.
            if (mouse.buttons_pressed & (1 << 2)) != 0 {
                let _ = ctx.window.set_cursor_grab(true);
                grab_cursor_pos = mouse.physical_position;
                ctx.window.set_cursor_visible(false);
            }

            // When ending camera rotation, release the cursor.
            if (mouse.buttons_released & (1 << 2)) != 0 {
                let _ = ctx.window.set_cursor_grab(false);
                ctx.window.set_cursor_visible(true);
            }

            let input = keymap.map(&keyboard, ctx.dt_filtered);
            let move_vec = camera.final_transform.rotation
                * Vec3::new(input["move_right"], input["move_up"], -input["move_fwd"])
                    .clamp_length_max(1.0)
                * 4.0f32.powf(input["boost"]);

            if (mouse.buttons_held & (1 << 2)) != 0 {
                // While we're rotating, the cursor should not move, so that upon revealing it,
                // it will be where we started the rotation motion at.
                let _ = ctx
                    .window
                    .set_cursor_position(winit::dpi::PhysicalPosition::new(
                        grab_cursor_pos.x,
                        grab_cursor_pos.y,
                    ));

                let sensitivity = 0.1;
                camera
                    .driver_mut::<YawPitch>()
                    .rotate_yaw_pitch(-sensitivity * mouse.delta.x, -sensitivity * mouse.delta.y);
            }

            camera
                .driver_mut::<Position>()
                .translate(move_vec * ctx.dt_filtered * state.camera_speed);
            camera.update(ctx.dt_filtered);

            if keyboard.is_down(VirtualKeyCode::Z) {
                car_pos.x += mouse.delta.x / 100.0;
            }
            //car_rot += 0.5 * ctx.dt;
            ctx.world_renderer.set_instance_transform(
                car_inst,
                Affine3A::from_rotation_translation(Quat::from_rotation_y(car_rot), car_pos),
            );

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

            if keyboard.was_just_pressed(VirtualKeyCode::L) {
                state.enable_emissive = !state.enable_emissive;
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
                          reset_path_tracer = true;
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
            if (sun_direction.dot(sun_direction_interp) - 1.0).abs() > 1e-5 {
                reset_path_tracer = true;
            }

            let sun_interp_t = if ctx.world_renderer.render_mode == RenderMode::Reference {
                1.0
            } else {
                (-1.0 * state.sun_rotation_smoothness).exp2()
            };

            sun_direction_interp =
                Vec3::lerp(sun_direction_interp, sun_direction, sun_interp_t).normalize();

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

            state.camera_position = camera.final_transform.position;
            state.camera_rotation = camera.final_transform.rotation;

            if keyboard.was_just_pressed(VirtualKeyCode::Tab) {
                show_gui = !show_gui;
            }

            if keyboard.was_just_pressed(VirtualKeyCode::C) {
                println!(
                    "position: {}, look_at: {}",
                    state.camera_position,
                    state.camera_position + state.camera_rotation * -Vec3::Z,
                );
            }

            ctx.world_renderer.rg_debug_hook = locked_rg_debug_hook.clone();

            if show_gui {
                ctx.imgui.take().unwrap().frame(|ui| {
                    if imgui::CollapsingHeader::new(im_str!("Tweaks"))
                        .default_open(true)
                        .build(ui)
                    {
                        imgui::Drag::<f32>::new(im_str!("EV shift"))
                            .range(-8.0..=12.0)
                            .speed(0.01)
                            .build(ui, &mut state.ev_shift);

                        ui.checkbox(
                            im_str!("Use dynamic exposure"),
                            &mut state.use_dynamic_exposure,
                        );

                        imgui::Drag::<f32>::new(im_str!("Emissive multiplier"))
                            .range(0.0..=10.0)
                            .speed(0.1)
                            .build(ui, &mut state.emissive_multiplier);

                        ui.checkbox(im_str!("Enable emissive"), &mut state.enable_emissive);

                        imgui::Drag::<f32>::new(im_str!("Light intensity multiplier"))
                            .range(0.0..=1000.0)
                            .speed(1.0)
                            .build(ui, &mut state.lights.multiplier);

                        imgui::Drag::<f32>::new(im_str!("Camera speed"))
                            .range(0.0..=10.0)
                            .speed(0.025)
                            .build(ui, &mut state.camera_speed);

                        imgui::Drag::<f32>::new(im_str!("Camera smoothness"))
                            .range(0.0..=20.0)
                            .speed(0.1)
                            .build(ui, &mut state.camera_smoothness);

                        imgui::Drag::<f32>::new(im_str!("Sun rotation smoothness"))
                            .range(0.0..=20.0)
                            .speed(0.1)
                            .build(ui, &mut state.sun_rotation_smoothness);

                        imgui::Drag::<f32>::new(im_str!("Field of view"))
                            .range(1.0..=120.0)
                            .speed(0.25)
                            .build(ui, &mut state.vertical_fov);

                        imgui::Drag::<f32>::new(im_str!("Sun size"))
                            .range(0.0..=10.0)
                            .speed(0.02)
                            .build(ui, &mut ctx.world_renderer.sun_size_multiplier);

                        /*ui.checkbox(
                            im_str!("Show world radiance cache"),
                            &mut ctx.world_renderer.debug_show_wrc,
                        );*/

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

                        ui.checkbox(
                            im_str!("Scroll irradiance cache"),
                            &mut ctx.world_renderer.ircache.enable_scroll,
                        );

                        imgui::Drag::<u32>::new(im_str!("GI spatial reuse passes"))
                            .range(1..=3)
                            .build(ui, &mut ctx.world_renderer.rtdgi.spatial_reuse_pass_count);

                        #[cfg(feature = "dlss")]
                        {
                            ui.checkbox(im_str!("Use DLSS"), &mut ctx.world_renderer.use_dlss);
                        }
                    }

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

                        /*if ui.radio_button_bool(
                            im_str!("World radiance cache"),
                            ctx.world_renderer.debug_mode == RenderDebugMode::WorldRadianceCache,
                        ) {
                            ctx.world_renderer.debug_mode = RenderDebugMode::WorldRadianceCache;
                        }*/

                        imgui::ComboBox::new(im_str!("Shading")).build_simple_string(
                            ui,
                            &mut ctx.world_renderer.debug_shading_mode,
                            &[
                                im_str!("Default"),
                                im_str!("No base color"),
                                im_str!("Diffuse GI"),
                                im_str!("Reflections"),
                                im_str!("RTX OFF"),
                                im_str!("Irradiance cache"),
                            ],
                        );

                        imgui::Drag::<u32>::new(im_str!("Max FPS"))
                            .range(1..=MAX_FPS_LIMIT)
                            .build(ui, &mut max_fps);

                        ui.checkbox(im_str!("Allow pass overlap"), unsafe {
                            &mut kajiya::rg::RG_ALLOW_PASS_OVERLAP
                        });
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

            let emissive_toggle_mult = if state.enable_emissive { 1.0 } else { 0.0 };
            for inst in &render_instances {
                ctx.world_renderer
                    .get_instance_dynamic_parameters_mut(*inst)
                    .emissive_multiplier = state.emissive_multiplier * emissive_toggle_mult;
            }

            ctx.world_renderer.ev_shift = state.ev_shift;
            ctx.world_renderer.dynamic_exposure.enabled = state.use_dynamic_exposure;

            if state.should_reset_path_tracer(&prev_state) {
                reset_path_tracer = true;
            }

            // Reset accumulation of the path tracer whenever the camera moves
            if (reset_path_tracer || keyboard.was_just_pressed(VirtualKeyCode::Back))
                && ctx.world_renderer.render_mode == RenderMode::Reference
            {
                ctx.world_renderer.reset_reference_accumulation = true;
                reset_path_tracer = false;
            }

            let lens = CameraLens {
                aspect_ratio: ctx.aspect_ratio(),
                vertical_fov: state.vertical_fov,
                ..Default::default()
            };

            WorldFrameDesc {
                camera_matrices: camera
                    .final_transform
                    .into_position_rotation()
                    .through(&lens),
                render_extent: ctx.render_extent,
                sun_direction: sun_direction_interp,
            }
        })?;
    }

    ron::ser::to_writer_pretty(
        File::create(APP_STATE_CONFIG_FILE_PATH)?,
        &state,
        Default::default(),
    )?;

    Ok(())
}
