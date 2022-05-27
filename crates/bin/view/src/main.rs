mod opt;

use opt::*;

use anyhow::Context;

use dolly::prelude::*;
use imgui::im_str;
use kajiya::{
    rg::GraphDebugHook,
    world_renderer::{AddMeshOptions, InstanceHandle, MeshHandle, WorldRenderer},
};
use kajiya_simple::*;

use std::{collections::HashMap, fs::File, path::PathBuf};
use structopt::StructOpt;

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

fn default_camera_speed() -> f32 {
    2.5
}

fn default_camera_smoothness() -> f32 {
    1.0
}

fn default_sun_rotation_smoothness() -> f32 {
    1.0
}

trait ShouldResetPathTracer {
    fn should_reset_path_tracer(&self, other: &Self) -> bool;
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct CameraState {
    position: Vec3,
    rotation: Quat,
    vertical_fov: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: Vec3::ONE,
            rotation: Quat::IDENTITY,
            vertical_fov: 62.0,
        }
    }
}

impl ShouldResetPathTracer for CameraState {
    fn should_reset_path_tracer(&self, other: &Self) -> bool {
        !self.position.abs_diff_eq(other.position, 1e-5)
            || !self.rotation.abs_diff_eq(other.rotation, 1e-5)
            || self.vertical_fov != other.vertical_fov
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct LightState {
    emissive_multiplier: f32,
    #[serde(default = "always_true")]
    enable_emissive: bool,
    sun: SunState,
    local_lights: LocalLightsState,
}

impl Default for LightState {
    fn default() -> Self {
        Self {
            emissive_multiplier: 1.0,
            enable_emissive: true,
            sun: SunState {
                theta: -4.54,
                phi: 1.48,
            },
            local_lights: LocalLightsState {
                theta: 1.0,
                phi: 1.0,
                count: 0,
                distance: 1.5,
                multiplier: 10.0,
            },
        }
    }
}

impl ShouldResetPathTracer for LightState {
    fn should_reset_path_tracer(&self, other: &Self) -> bool {
        self.emissive_multiplier != other.emissive_multiplier
            || self.sun != other.sun
            || self.local_lights != other.local_lights
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct MovementState {
    #[serde(default = "default_camera_speed")]
    camera_speed: f32,
    #[serde(default = "default_camera_smoothness")]
    camera_smoothness: f32,
    #[serde(default = "default_sun_rotation_smoothness")]
    sun_rotation_smoothness: f32,
}

impl Default for MovementState {
    fn default() -> Self {
        Self {
            camera_speed: 1.0,
            camera_smoothness: default_camera_smoothness(),
            sun_rotation_smoothness: default_sun_rotation_smoothness(),
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ExposureState {
    ev_shift: f32,
    #[serde(default)]
    use_dynamic_adaptation: bool,
}

impl Default for ExposureState {
    fn default() -> Self {
        Self {
            ev_shift: 0.0,
            use_dynamic_adaptation: true,
        }
    }
}

struct RuntimeState {
    camera: CameraRig,
    mouse: MouseState,
    keyboard: KeyboardState,
    keymap: KeyboardMap,

    render_instances: Vec<InstanceHandle>,

    show_gui: bool,
    sun_direction_interp: Vec3,
    left_click_edit_mode: LeftClickEditMode,

    max_fps: u32,
    locked_rg_debug_hook: Option<GraphDebugHook>,
    grab_cursor_pos: winit::dpi::PhysicalPosition<f64>,

    reset_path_tracer: bool,
}

impl RuntimeState {
    fn load_scene(
        &mut self,
        world_renderer: &mut WorldRenderer,
        scene_name: &str,
    ) -> anyhow::Result<()> {
        let scene_file = format!("assets/scenes/{}.ron", scene_name);
        let scene_desc: SceneDesc = ron::de::from_reader(
            File::open(&scene_file)
                .with_context(|| format!("Opening scene file {}", scene_file))?,
        )?;

        let mut known_meshes: HashMap<PathBuf, MeshHandle> = HashMap::new();

        for instance in scene_desc.instances {
            let path = PathBuf::from(format!("/baked/{}.mesh", instance.mesh));

            let mesh = *known_meshes.entry(path.clone()).or_insert_with(|| {
                world_renderer
                    .add_baked_mesh(path, AddMeshOptions::new())
                    .unwrap()
            });

            self.render_instances.push(world_renderer.add_instance(
                mesh,
                Affine3A::from_scale_rotation_translation(
                    instance.scale.into(),
                    Quat::IDENTITY,
                    instance.position.into(),
                ),
            ));
        }

        /*let mut test_obj_pos = Vec3::Y * -0.01;
        let mut test_obj_rot = 0.0f32;
        let test_obj_inst = if USE_TEST_DYNAMIC_OBJECT {
            let test_obj_mesh =
                world_renderer.add_baked_mesh("/baked/336_lrm.mesh", AddMeshOptions::default())?;
            Some(world_renderer.add_instance(
                test_obj_mesh,
                Affine3A::from_rotation_translation(Quat::IDENTITY, test_obj_pos),
            ))
        } else {
            None
        };*/

        Ok(())
    }

    fn frame(&mut self, mut ctx: FrameContext, persisted: &mut PersistedState) -> WorldFrameDesc {
        // Limit framerate. Not particularly precise.
        if self.max_fps != MAX_FPS_LIMIT {
            std::thread::sleep(std::time::Duration::from_micros(
                1_000_000 / self.max_fps as u64,
            ));
        }

        let smooth = self.camera.driver_mut::<Smooth>();
        if ctx.world_renderer.render_mode == RenderMode::Reference {
            smooth.position_smoothness = 0.0;
            smooth.rotation_smoothness = 0.0;
        } else {
            smooth.position_smoothness = persisted.movement.camera_smoothness;
            smooth.rotation_smoothness = persisted.movement.camera_smoothness;
        }

        self.keyboard.update(ctx.events);
        self.mouse.update(ctx.events);

        // When starting camera rotation, hide the mouse cursor, and capture it to the window.
        if (self.mouse.buttons_pressed & (1 << 2)) != 0 {
            let _ = ctx.window.set_cursor_grab(true);
            self.grab_cursor_pos = self.mouse.physical_position;
            ctx.window.set_cursor_visible(false);
        }

        // When ending camera rotation, release the cursor.
        if (self.mouse.buttons_released & (1 << 2)) != 0 {
            let _ = ctx.window.set_cursor_grab(false);
            ctx.window.set_cursor_visible(true);
        }

        let input = self.keymap.map(&self.keyboard, ctx.dt_filtered);
        let move_vec = self.camera.final_transform.rotation
            * Vec3::new(input["move_right"], input["move_up"], -input["move_fwd"])
                .clamp_length_max(1.0)
            * 4.0f32.powf(input["boost"]);

        if (self.mouse.buttons_held & (1 << 2)) != 0 {
            // While we're rotating, the cursor should not move, so that upon revealing it,
            // it will be where we started the rotation motion at.
            let _ = ctx
                .window
                .set_cursor_position(winit::dpi::PhysicalPosition::new(
                    self.grab_cursor_pos.x,
                    self.grab_cursor_pos.y,
                ));

            let sensitivity = 0.1;
            self.camera.driver_mut::<YawPitch>().rotate_yaw_pitch(
                -sensitivity * self.mouse.delta.x,
                -sensitivity * self.mouse.delta.y,
            );
        }

        self.camera
            .driver_mut::<Position>()
            .translate(move_vec * ctx.dt_filtered * persisted.movement.camera_speed);
        self.camera.update(ctx.dt_filtered);

        /*if self.keyboard.is_down(VirtualKeyCode::Z) {
            test_obj_pos.x += mouse.delta.x / 100.0;
        }

        if SPIN_TEST_DYNAMIC_OBJECT {
            test_obj_rot += 0.5 * ctx.dt_filtered;
        }

        if let Some(test_obj_inst) = test_obj_inst {
            ctx.world_renderer.set_instance_transform(
                test_obj_inst,
                Affine3A::from_rotation_translation(
                    Quat::from_rotation_y(test_obj_rot),
                    test_obj_pos,
                ),
            );
        }*/

        if self.keyboard.was_just_pressed(VirtualKeyCode::Space) {
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

        if self.keyboard.was_just_pressed(VirtualKeyCode::L) {
            persisted.light.enable_emissive = !persisted.light.enable_emissive;
        }

        /*if state.keyboard.was_just_pressed(VirtualKeyCode::Delete) {
            if let Some(persisted_app_state) = persisted_app_state.as_ref() {
                *state = persisted_app_state.clone();

                camera
                    .driver_mut::<YawPitch>()
                    .set_rotation_quat(state.camera_rotation);
                camera.driver_mut::<Position>().position = state.camera_position;
            }
        }*/

        if self.mouse.buttons_held & 1 != 0 {
            let theta_delta =
                (self.mouse.delta.x / ctx.render_extent[0] as f32) * -std::f32::consts::TAU;
            let phi_delta =
                (self.mouse.delta.y / ctx.render_extent[1] as f32) * std::f32::consts::PI;

            match self.left_click_edit_mode {
                LeftClickEditMode::MoveSun => {
                    persisted.light.sun.theta += theta_delta;
                    persisted.light.sun.phi += phi_delta;
                } /*LeftClickEditMode::MoveLocalLights => {
                      persisted.light.lights.theta += theta_delta;
                      persisted.light.lights.phi += phi_delta;
                  }*/
            }
        }

        /*if self.keyboard.is_down(VirtualKeyCode::Z) {
            persisted.light.local_lights.distance /= 0.99;
        }
        if self.keyboard.is_down(VirtualKeyCode::X) {
            persisted.light.local_lights.distance *= 0.99;
        }*/

        //state.sun.phi += dt;
        //state.sun.phi %= std::f32::consts::TAU;

        let sun_direction = persisted.light.sun.direction();
        if (sun_direction.dot(self.sun_direction_interp) - 1.0).abs() > 1e-5 {
            self.reset_path_tracer = true;
        }

        let sun_interp_t = if ctx.world_renderer.render_mode == RenderMode::Reference {
            1.0
        } else {
            (-1.0 * persisted.movement.sun_rotation_smoothness).exp2()
        };

        self.sun_direction_interp =
            Vec3::lerp(self.sun_direction_interp, sun_direction, sun_interp_t).normalize();

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

        persisted.camera.position = self.camera.final_transform.position;
        persisted.camera.rotation = self.camera.final_transform.rotation;

        if self.keyboard.was_just_pressed(VirtualKeyCode::Tab) {
            self.show_gui = !self.show_gui;
        }

        if self.keyboard.was_just_pressed(VirtualKeyCode::C) {
            println!(
                "position: {}, look_at: {}",
                persisted.camera.position,
                persisted.camera.position + persisted.camera.rotation * -Vec3::Z,
            );
        }

        ctx.world_renderer.rg_debug_hook = self.locked_rg_debug_hook.clone();

        if self.show_gui {
            ctx.imgui.take().unwrap().frame(|ui| {
                if imgui::CollapsingHeader::new(im_str!("Tweaks"))
                    .default_open(true)
                    .build(ui)
                {
                    imgui::Drag::<f32>::new(im_str!("EV shift"))
                        .range(-8.0..=12.0)
                        .speed(0.01)
                        .build(ui, &mut persisted.exposure.ev_shift);

                    ui.checkbox(
                        im_str!("Use dynamic exposure"),
                        &mut persisted.exposure.use_dynamic_adaptation,
                    );

                    imgui::Drag::<f32>::new(im_str!("Emissive multiplier"))
                        .range(0.0..=10.0)
                        .speed(0.1)
                        .build(ui, &mut persisted.light.emissive_multiplier);

                    ui.checkbox(
                        im_str!("Enable emissive"),
                        &mut persisted.light.enable_emissive,
                    );

                    imgui::Drag::<f32>::new(im_str!("Light intensity multiplier"))
                        .range(0.0..=1000.0)
                        .speed(1.0)
                        .build(ui, &mut persisted.light.local_lights.multiplier);

                    imgui::Drag::<f32>::new(im_str!("Camera speed"))
                        .range(0.0..=10.0)
                        .speed(0.025)
                        .build(ui, &mut persisted.movement.camera_speed);

                    imgui::Drag::<f32>::new(im_str!("Camera smoothness"))
                        .range(0.0..=20.0)
                        .speed(0.1)
                        .build(ui, &mut persisted.movement.camera_smoothness);

                    imgui::Drag::<f32>::new(im_str!("Sun rotation smoothness"))
                        .range(0.0..=20.0)
                        .speed(0.1)
                        .build(ui, &mut persisted.movement.sun_rotation_smoothness);

                    imgui::Drag::<f32>::new(im_str!("Field of view"))
                        .range(1.0..=120.0)
                        .speed(0.25)
                        .build(ui, &mut persisted.camera.vertical_fov);

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
                        .build(ui, &mut self.max_fps);

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

                        let style = self.locked_rg_debug_hook.as_ref().and_then(|hook| {
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
                            ctx.world_renderer.rg_debug_hook = Some(kajiya::rg::GraphDebugHook {
                                render_scope: scope.clone(),
                            });

                            if ui.is_item_clicked(imgui::MouseButton::Left) {
                                if self.locked_rg_debug_hook == ctx.world_renderer.rg_debug_hook {
                                    self.locked_rg_debug_hook = None;
                                } else {
                                    self.locked_rg_debug_hook =
                                        ctx.world_renderer.rg_debug_hook.clone();
                                }
                            }
                        }
                    }
                }
            });
        }

        let emissive_toggle_mult = if persisted.light.enable_emissive {
            1.0
        } else {
            0.0
        };

        for inst in &self.render_instances {
            ctx.world_renderer
                .get_instance_dynamic_parameters_mut(*inst)
                .emissive_multiplier = persisted.light.emissive_multiplier * emissive_toggle_mult;
        }

        ctx.world_renderer.ev_shift = persisted.exposure.ev_shift;
        ctx.world_renderer.dynamic_exposure.enabled = persisted.exposure.use_dynamic_adaptation;

        // TODO
        /*if state.should_reset_path_tracer(&prev_state) {
            reset_path_tracer = true;
        }*/

        // Reset accumulation of the path tracer whenever the camera moves
        if (self.reset_path_tracer || self.keyboard.was_just_pressed(VirtualKeyCode::Back))
            && ctx.world_renderer.render_mode == RenderMode::Reference
        {
            ctx.world_renderer.reset_reference_accumulation = true;
            self.reset_path_tracer = false;
        }

        let lens = CameraLens {
            aspect_ratio: ctx.aspect_ratio(),
            vertical_fov: persisted.camera.vertical_fov,
            ..Default::default()
        };

        WorldFrameDesc {
            camera_matrices: self
                .camera
                .final_transform
                .into_position_rotation()
                .through(&lens),
            render_extent: ctx.render_extent,
            sun_direction: self.sun_direction_interp,
        }
    }
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct PersistedState {
    camera: CameraState,
    exposure: ExposureState,
    light: LightState,
    movement: MovementState,
}

struct AppState {
    persisted: PersistedState,
    runtime: RuntimeState,
    kajiya: SimpleMainLoop,
}

const MAX_FPS_LIMIT: u32 = 256;

impl AppState {
    fn new(persisted: PersistedState, opt: &Opt) -> anyhow::Result<Self> {
        let kajiya = SimpleMainLoop::builder()
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

        let camera: CameraRig = CameraRig::builder()
            .with(Position::new(persisted.camera.position))
            .with(YawPitch::new().rotation_quat(persisted.camera.rotation))
            .with(Smooth::default())
            .build();

        // Mitsuba match
        /*let mut camera = camera::FirstPersonCamera::new(Vec3::new(-2.0, 4.0, 8.0));
        camera.fov = 35.0 * 9.0 / 16.0;
        camera.look_at(Vec3::new(0.0, 0.75, 0.0));*/

        let mouse: MouseState = Default::default();
        let keyboard: KeyboardState = Default::default();

        let keymap = KeyboardMap::new()
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

        let sun_direction_interp = persisted.light.sun.direction();

        Ok(Self {
            persisted,
            runtime: RuntimeState {
                camera,
                mouse,
                keyboard,
                keymap,

                render_instances: vec![],

                show_gui: false,
                sun_direction_interp,
                left_click_edit_mode: LeftClickEditMode::MoveSun,

                max_fps: MAX_FPS_LIMIT,
                locked_rg_debug_hook: None,
                grab_cursor_pos: Default::default(),

                reset_path_tracer: false,
            },
            kajiya,
        })
    }

    fn load_scene(&mut self, scene_name: &str) -> anyhow::Result<()> {
        self.runtime
            .load_scene(&mut self.kajiya.world_renderer, scene_name)
    }

    fn run(self) -> anyhow::Result<()> {
        let Self {
            mut persisted,
            mut runtime,
            kajiya,
        } = self;

        kajiya.run(|ctx| runtime.frame(ctx, &mut persisted))
    }
}

#[derive(PartialEq, Eq)]
enum LeftClickEditMode {
    MoveSun,
    //MoveLocalLights,
}

const APP_STATE_CONFIG_FILE_PATH: &str = "view_state.ron";

// If true, have an additional test/debug object in the scene, whose position can be changed by holding `Z` and moving the mouse.
const USE_TEST_DYNAMIC_OBJECT: bool = false;
const SPIN_TEST_DYNAMIC_OBJECT: bool = false;

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();
    let persisted: PersistedState = Default::default();

    let mut state = AppState::new(persisted, &opt)?;

    state.load_scene(&opt.scene)?;
    state.run()?;

    /*ron::ser::to_writer_pretty(
        File::create(APP_STATE_CONFIG_FILE_PATH)?,
        &state,
        Default::default(),
    )?;*/

    Ok(())
}
