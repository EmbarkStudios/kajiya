use anyhow::Context;

use dolly::prelude::*;
use kajiya::{
    rg::GraphDebugHook,
    world_renderer::{AddMeshOptions, InstanceHandle, MeshHandle, WorldRenderer},
};
use kajiya_simple::*;

use crate::{opt::Opt, scene::SceneDesc, PersistedState};

use std::{collections::HashMap, fs::File, path::PathBuf};

pub const MAX_FPS_LIMIT: u32 = 256;

pub struct RuntimeState {
    pub camera: CameraRig,
    pub mouse: MouseState,
    pub keyboard: KeyboardState,
    pub keymap: KeyboardMap,

    pub render_instances: Vec<InstanceHandle>,

    pub show_gui: bool,
    pub sun_direction_interp: Vec3,
    pub left_click_edit_mode: LeftClickEditMode,

    pub max_fps: u32,
    pub locked_rg_debug_hook: Option<GraphDebugHook>,
    pub grab_cursor_pos: winit::dpi::PhysicalPosition<f64>,

    pub reset_path_tracer: bool,
}

impl RuntimeState {
    pub fn new(persisted: &PersistedState, _opt: &Opt) -> Self {
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

        Self {
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
        }
    }

    pub fn load_scene(
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

    fn update_camera(&mut self, persisted: &mut PersistedState, ctx: &FrameContext) {
        let smooth = self.camera.driver_mut::<Smooth>();
        if ctx.world_renderer.render_mode == RenderMode::Reference {
            smooth.position_smoothness = 0.0;
            smooth.rotation_smoothness = 0.0;
        } else {
            smooth.position_smoothness = persisted.movement.camera_smoothness;
            smooth.rotation_smoothness = persisted.movement.camera_smoothness;
        }

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

        persisted.camera.position = self.camera.final_transform.position;
        persisted.camera.rotation = self.camera.final_transform.rotation;

        if self.keyboard.was_just_pressed(VirtualKeyCode::C) {
            println!(
                "position: {}, look_at: {}",
                persisted.camera.position,
                persisted.camera.position + persisted.camera.rotation * -Vec3::Z,
            );
        }
    }

    fn update_sun(&mut self, persisted: &mut PersistedState, ctx: &FrameContext) {
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
    }

    fn update_lights(&mut self, persisted: &mut PersistedState, ctx: &mut FrameContext) {
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

        /*if self.keyboard.is_down(VirtualKeyCode::Z) {
            persisted.light.local_lights.distance /= 0.99;
        }
        if self.keyboard.is_down(VirtualKeyCode::X) {
            persisted.light.local_lights.distance *= 0.99;
        }*/

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
    }

    fn update_objects(&mut self, persisted: &mut PersistedState, ctx: &mut FrameContext) {
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
    }

    pub fn frame(
        &mut self,
        mut ctx: FrameContext,
        persisted: &mut PersistedState,
    ) -> WorldFrameDesc {
        // Limit framerate. Not particularly precise.
        if self.max_fps != MAX_FPS_LIMIT {
            std::thread::sleep(std::time::Duration::from_micros(
                1_000_000 / self.max_fps as u64,
            ));
        }

        self.keyboard.update(ctx.events);
        self.mouse.update(ctx.events);

        /*if state.keyboard.was_just_pressed(VirtualKeyCode::Delete) {
            if let Some(persisted_app_state) = persisted_app_state.as_ref() {
                *state = persisted_app_state.clone();

                camera
                    .driver_mut::<YawPitch>()
                    .set_rotation_quat(state.camera_rotation);
                camera.driver_mut::<Position>().position = state.camera_position;
            }
        }*/

        self.do_gui(persisted, &mut ctx);
        self.update_lights(persisted, &mut ctx);
        self.update_objects(persisted, &mut ctx);

        self.update_camera(persisted, &ctx);
        self.update_sun(persisted, &ctx);

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

#[derive(PartialEq, Eq)]
pub enum LeftClickEditMode {
    MoveSun,
    //MoveLocalLights,
}
