use imgui::im_str;
use kajiya::RenderOverrideFlags;
use kajiya_simple::*;

use crate::{
    runtime::{RuntimeState, MAX_FPS_LIMIT},
    PersistedState,
};

impl RuntimeState {
    pub fn do_gui(&mut self, persisted: &mut PersistedState, ctx: &mut FrameContext) {
        if self.keyboard.was_just_pressed(VirtualKeyCode::Tab) {
            self.show_gui = !self.show_gui;
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

                    imgui::Drag::<f32>::new(im_str!("Adaptation speed"))
                        .range(-4.0..=4.0)
                        .speed(0.01)
                        .build(ui, &mut persisted.exposure.dynamic_adaptation_speed);

                    imgui::Drag::<f32>::new(im_str!("Luminance histogram low clip"))
                        .range(0.0..=1.0)
                        .speed(0.001)
                        .build(ui, &mut persisted.exposure.dynamic_adaptation_low_clip);

                    imgui::Drag::<f32>::new(im_str!("Luminance histogram high clip"))
                        .range(0.0..=1.0)
                        .speed(0.001)
                        .build(ui, &mut persisted.exposure.dynamic_adaptation_high_clip);

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
                        .build(ui, &mut persisted.light.sun.size_multiplier);

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

                    ui.checkbox(
                        im_str!("Allow diffuse ray reuse for rtr"),
                        &mut ctx.world_renderer.rtr.reuse_rtdgi_rays,
                    );

                    #[cfg(feature = "dlss")]
                    {
                        ui.checkbox(im_str!("Use DLSS"), &mut ctx.world_renderer.use_dlss);
                    }
                }

                if imgui::CollapsingHeader::new(im_str!("Overrides"))
                    .default_open(false)
                    .build(ui)
                {
                    macro_rules! do_flag {
                        ($flag:path, $name:literal) => {
                            let mut is_set: bool =
                                ctx.world_renderer.render_overrides.has_flag($flag);
                            ui.checkbox(im_str!($name), &mut is_set);
                            ctx.world_renderer.render_overrides.set_flag($flag, is_set);
                        };
                    }

                    do_flag!(
                        RenderOverrideFlags::FORCE_FACE_NORMALS,
                        "Force face normals"
                    );
                    do_flag!(RenderOverrideFlags::NO_NORMAL_MAPS, "No normal maps");
                    do_flag!(
                        RenderOverrideFlags::FLIP_NORMAL_MAP_YZ,
                        "Flip normal map YZ"
                    );
                    do_flag!(RenderOverrideFlags::NO_METAL, "No metal");

                    imgui::Drag::<f32>::new(im_str!("Roughness scale"))
                        .range(0.0..=4.0)
                        .speed(0.001)
                        .build(
                            ui,
                            &mut ctx.world_renderer.render_overrides.material_roughness_scale,
                        );
                }

                if imgui::CollapsingHeader::new(im_str!("Sequence"))
                    .default_open(false)
                    .build(ui)
                {
                    if ui.button(im_str!("Add key"), [0.0, 0.0]) {
                        self.add_sequence_keyframe(persisted);
                    }

                    ui.same_line(0.0);
                    if self.is_sequence_playing() {
                        if ui.button(im_str!("Stop"), [0.0, 0.0]) {
                            self.stop_sequence();
                        }
                    } else if ui.button(im_str!("Play"), [0.0, 0.0]) {
                        self.play_sequence(persisted);
                    }

                    ui.same_line(0.0);
                    ui.set_next_item_width(60.0);
                    imgui::Drag::<f32>::new(im_str!("Speed"))
                        .range(0.0..=4.0)
                        .speed(0.01)
                        .build(ui, &mut self.sequence_playback_speed);

                    if self.active_camera_key.is_some() {
                        ui.same_line(0.0);
                        if ui.button(im_str!("Deselect key"), [0.0, 0.0]) {
                            self.active_camera_key = None;
                        }
                    }

                    enum Cmd {
                        JumpToKey(usize),
                        DeleteKey(usize),
                        ReplaceKey(usize),
                        None,
                    }
                    let mut cmd = Cmd::None;

                    persisted.sequence.each_key(|i, item| {
                        let active = Some(i) == self.active_camera_key;

                        let label = if active {
                            im_str!("-> {}:", i)
                        } else {
                            im_str!("{}:", i)
                        };

                        if ui.button(&label, [0.0, 0.0]) {
                            cmd = Cmd::JumpToKey(i);
                        }

                        ui.same_line(0.0);
                        ui.set_next_item_width(60.0);
                        imgui::InputFloat::new(ui, &im_str!("duration##{}", i), &mut item.duration)
                            .build();

                        ui.same_line(0.0);
                        ui.checkbox(
                            &im_str!("Pos##{}", i),
                            &mut item.value.camera_position.is_some,
                        );

                        ui.same_line(0.0);
                        ui.checkbox(
                            &im_str!("Dir##{}", i),
                            &mut item.value.camera_direction.is_some,
                        );

                        ui.same_line(0.0);
                        ui.checkbox(&im_str!("Sun##{}", i), &mut item.value.towards_sun.is_some);

                        ui.same_line(0.0);
                        if ui.button(&im_str!("Delete##{}", i), [0.0, 0.0]) {
                            cmd = Cmd::DeleteKey(i);
                        }

                        ui.same_line(0.0);
                        if ui.button(&im_str!("Replace##{}:", i), [0.0, 0.0]) {
                            cmd = Cmd::ReplaceKey(i);
                        }
                    });

                    match cmd {
                        Cmd::JumpToKey(i) => self.jump_to_sequence_key(persisted, i),
                        Cmd::DeleteKey(i) => self.delete_camera_sequence_key(persisted, i),
                        Cmd::ReplaceKey(i) => self.replace_camera_sequence_key(persisted, i),
                        Cmd::None => {}
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
    }
}
