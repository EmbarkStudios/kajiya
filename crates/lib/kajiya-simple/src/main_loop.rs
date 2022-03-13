use std::collections::VecDeque;

#[cfg(feature = "use-egui")]
use egui::{
    style::{Selection, WidgetVisuals, Widgets},
    Color32, Context, Modifiers, Rounding, Stroke, TextStyle, Vec2,
};

use kajiya::{
    backend::{vulkan::RenderBackendConfig, *},
    frame_desc::WorldFrameDesc,
    rg,
    ui_renderer::UiRenderer,
    world_renderer::WorldRenderer,
};

#[cfg(feature = "dear-imgui")]
use kajiya_imgui::ImGuiBackend;

#[cfg(feature = "use-egui")]
use kajiya_egui::{EguiBackend, EguiState};

use turbosloth::*;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::{Fullscreen, WindowBuilder},
};

#[cfg(feature = "use-egui")]
use crate::MouseState;

#[cfg(feature = "use-egui")]
const MOUSE_BUTTON_LEFT_PRESSED: u32 = 1;
#[cfg(feature = "use-egui")]
const MOUSE_BUTTON_LEFT_RELEASED: u32 = 1;

pub struct FrameContext<'a> {
    pub dt_filtered: f32,
    pub render_extent: [u32; 2],
    pub events: &'a [WindowEvent<'static>],
    pub world_renderer: &'a mut WorldRenderer,

    #[cfg(feature = "dear-imgui")]
    pub imgui: Option<ImguiContext<'a>>,

    #[cfg(feature = "use-egui")]
    pub egui: Option<EguiContext<'a>>,
}

impl<'a> FrameContext<'a> {
    pub fn aspect_ratio(&self) -> f32 {
        self.render_extent[0] as f32 / self.render_extent[1] as f32
    }
}

#[cfg(feature = "dear-imgui")]
pub struct ImguiContext<'a> {
    imgui: &'a mut imgui::Context,
    imgui_backend: &'a mut ImGuiBackend,
    ui_renderer: &'a mut UiRenderer,
    window: &'a winit::window::Window,
    dt_filtered: f32,
}

#[cfg(feature = "use-egui")]
pub struct EguiContext<'a> {
    egui: &'a mut EguiState,
    egui_backend: &'a mut EguiBackend,
    ui_renderer: &'a mut UiRenderer,
    dt_filtered: f32,
}

#[cfg(feature = "use-egui")]
impl<'a> EguiContext<'a> {
    pub fn ctx(&self) -> &Context {
        &self.egui.egui_context
    }

    fn process_input(&mut self, mouse: &MouseState) {
        let mut mouse_position = (
            mouse.physical_position.x as f32,
            mouse.physical_position.y as f32,
        );

        mouse_position.0 /= self.egui.raw_input.pixels_per_point.unwrap();
        mouse_position.1 /= self.egui.raw_input.pixels_per_point.unwrap();

        self.egui.last_mouse_pos = Some(mouse_position);

        self.egui
            .raw_input
            .events
            .push(egui::Event::PointerMoved(egui::pos2(
                mouse_position.0,
                mouse_position.1,
            )));

        let pos = egui::pos2(mouse_position.0, mouse_position.1);

        if mouse.buttons_pressed == MOUSE_BUTTON_LEFT_PRESSED {
            self.egui.raw_input.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: Modifiers::default(),
            });
        }

        if mouse.buttons_released == MOUSE_BUTTON_LEFT_RELEASED {
            self.egui.raw_input.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: Modifiers::default(),
            });
        }

        if mouse.wheel_delta != 0.0 {
            let scroll_delta = Vec2::new(0.0, mouse.wheel_delta);
            self.egui
                .raw_input
                .events
                .push(egui::Event::Scroll(scroll_delta));
        }
    }

    pub fn frame(&mut self, mouse: &MouseState, callback: impl FnOnce(&Context)) {
        self.process_input(mouse);

        callback(&self.egui.egui_context);

        // Update delta time
        self.egui.last_dt = self.dt_filtered as f64;

        // Prepare the egui context's frame so that the renderer can finish frame
        EguiBackend::prepare_frame(&mut self.egui);

        // (Update input)...
        self.egui_backend.finish_frame(
            &mut self.egui.egui_context,
            self.egui.window_size,
            self.ui_renderer,
        );
    }

    pub fn get_theme_visuals() -> egui::style::Visuals {
        const WINDOW_BG_COLOR: Color32 = Color32::from_rgba_premultiplied(13, 13, 37, 150);
        const WINDOW_OUTLINE_COLOR: Color32 = Color32::from_rgba_premultiplied(37, 85, 136, 255);
        const WIDGET_BG_COLOR: Color32 = Color32::from_rgba_premultiplied(82, 42, 69, 255);
        const WIDGET_STROKE_FG_COLOR: Color32 = Color32::from_gray(240);
        const WIDGET_STROKE_BG_COLOR: Color32 = Color32::from_gray(150);
        const WIGDET_TEXT_COLOR: Color32 = Color32::from_rgba_premultiplied(206, 206, 206, 255);
        const WIGDET_HOVERED_COLOR: Color32 = Color32::from_rgba_premultiplied(104, 0, 98, 255);
        const ACTIVE_SELECTED_COLOR: Color32 = Color32::from_rgba_premultiplied(140, 0, 148, 255);
        const TEXT_EDIT_BG_COLOR: Color32 = Color32::from_rgba_premultiplied(11, 11, 17, 255);
        const SELECTED_ITEM_COLOR: Color32 = Color32::from_rgba_premultiplied(89, 57, 87, 255);
        const NORMAL_TEXT_COLOR: Color32 = Color32::WHITE;

        #[cfg(feature = "use-egui")]
        let visuals = egui::style::Visuals {
            widgets: Widgets {
                noninteractive: WidgetVisuals {
                    bg_fill: WINDOW_BG_COLOR,                          // window background
                    bg_stroke: Stroke::new(1.0, WINDOW_OUTLINE_COLOR), // separators, indentation lines, windows outlines
                    fg_stroke: Stroke::new(1.0, NORMAL_TEXT_COLOR),    // normal text color
                    rounding: Rounding::same(2.0),
                    expansion: 0.0,
                },
                inactive: WidgetVisuals {
                    bg_fill: WIDGET_BG_COLOR, // button, sliders background
                    bg_stroke: Default::default(),
                    fg_stroke: Stroke::new(1.0, WIGDET_TEXT_COLOR), // button text
                    rounding: Rounding::same(2.0),
                    expansion: 0.0,
                },
                hovered: WidgetVisuals {
                    bg_fill: WIGDET_HOVERED_COLOR,
                    bg_stroke: Stroke::new(1.0, WIDGET_STROKE_BG_COLOR), // e.g. hover over window edge or button
                    fg_stroke: Stroke::new(1.5, WIDGET_STROKE_FG_COLOR),
                    rounding: Rounding::same(3.0),
                    expansion: 1.0,
                },
                active: WidgetVisuals {
                    bg_fill: ACTIVE_SELECTED_COLOR,
                    bg_stroke: Stroke::new(1.0, NORMAL_TEXT_COLOR),
                    fg_stroke: Stroke::new(2.0, NORMAL_TEXT_COLOR),
                    rounding: Rounding::same(2.0),
                    expansion: 1.0,
                },
                ..Widgets::dark()
            },
            selection: Selection {
                bg_fill: SELECTED_ITEM_COLOR,
                ..Selection::default()
            },
            hyperlink_color: ACTIVE_SELECTED_COLOR,
            faint_bg_color: WINDOW_BG_COLOR,
            extreme_bg_color: TEXT_EDIT_BG_COLOR, // e.g. TextEdit background
            code_bg_color: TEXT_EDIT_BG_COLOR,
            ..egui::style::Visuals::dark()
        };

        visuals
    }
}

#[cfg(feature = "dear-imgui")]
impl<'a> ImguiContext<'a> {
    pub fn frame(self, callback: impl FnOnce(&imgui::Ui<'_>)) {
        let ui = self
            .imgui_backend
            .prepare_frame(self.window, self.imgui, self.dt_filtered);
        callback(&ui);
        self.imgui_backend
            .finish_frame(ui, self.window, self.ui_renderer);
    }
}

struct MainLoopOptional {
    #[cfg(feature = "dear-imgui")]
    imgui_backend: ImGuiBackend,

    #[cfg(feature = "dear-imgui")]
    imgui: imgui::Context,

    #[cfg(feature = "use-egui")]
    egui_backend: EguiBackend,

    #[cfg(feature = "use-egui")]
    egui: EguiState,

    #[cfg(feature = "puffin-server")]
    _puffin_server: puffin_http::Server,
}

pub enum WindowScale {
    Exact(f32),

    // Follow resolution scaling preferences in the OS
    SystemNative,
}

pub enum FullscreenMode {
    Borderless,

    /// Seems to be the only way for stutter-free rendering on Nvidia + Win10.
    Exclusive,
}

pub struct SimpleMainLoopBuilder {
    resolution: [u32; 2],
    vsync: bool,
    fullscreen: Option<FullscreenMode>,
    graphics_debugging: bool,
    default_log_level: log::LevelFilter,
    window_scale: WindowScale,
    temporal_upsampling: f32,
}

impl Default for SimpleMainLoopBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleMainLoopBuilder {
    pub fn new() -> Self {
        SimpleMainLoopBuilder {
            resolution: [1280, 720],
            vsync: true,
            fullscreen: None,
            graphics_debugging: false,
            default_log_level: log::LevelFilter::Warn,
            window_scale: WindowScale::SystemNative,
            temporal_upsampling: 1.0,
        }
    }

    pub fn resolution(mut self, resolution: [u32; 2]) -> Self {
        self.resolution = resolution;
        self
    }

    pub fn vsync(mut self, vsync: bool) -> Self {
        self.vsync = vsync;
        self
    }

    pub fn graphics_debugging(mut self, graphics_debugging: bool) -> Self {
        self.graphics_debugging = graphics_debugging;
        self
    }

    pub fn default_log_level(mut self, default_log_level: log::LevelFilter) -> Self {
        self.default_log_level = default_log_level;
        self
    }

    pub fn fullscreen(mut self, fullscreen: Option<FullscreenMode>) -> Self {
        self.fullscreen = fullscreen;
        self
    }

    // TODO; not hooked up yet
    pub fn window_scale(mut self, window_scale: WindowScale) -> Self {
        self.window_scale = window_scale;
        self
    }

    /// Must be >= 1.0. The rendering resolution will be 1.0 / `temporal_upsampling`,
    /// and will be upscaled to the target resolution by TAA. Greater values mean faster
    /// rendering, but temporal shimmering artifacts and blurriness.
    pub fn temporal_upsampling(mut self, temporal_upsampling: f32) -> Self {
        self.temporal_upsampling = temporal_upsampling.clamp(1.0, 8.0);
        self
    }

    pub fn build(self, window_builder: WindowBuilder) -> anyhow::Result<SimpleMainLoop> {
        SimpleMainLoop::build(self, window_builder)
    }
}

pub struct SimpleMainLoop {
    pub window: winit::window::Window,
    pub world_renderer: WorldRenderer,
    ui_renderer: UiRenderer,

    optional: MainLoopOptional,

    event_loop: EventLoop<()>,
    render_backend: RenderBackend,
    rg_renderer: kajiya::rg::renderer::Renderer,
    render_extent: [u32; 2],
}

impl SimpleMainLoop {
    pub fn builder() -> SimpleMainLoopBuilder {
        SimpleMainLoopBuilder::new()
    }

    fn build(
        builder: SimpleMainLoopBuilder,
        mut window_builder: WindowBuilder,
    ) -> anyhow::Result<Self> {
        kajiya::logging::set_up_logging(builder.default_log_level)?;
        std::env::set_var("SMOL_THREADS", "64"); // HACK; TODO: get a real executor

        // Note: asking for the logical size means that if the OS is using DPI scaling,
        // we'll get a physically larger window (with more pixels).
        // The internal rendering resolution will still be what was asked of the `builder`,
        // and the last blit pass will perform spatial upsampling.
        window_builder = window_builder.with_inner_size(winit::dpi::LogicalSize::new(
            builder.resolution[0] as f64,
            builder.resolution[1] as f64,
        ));

        let event_loop = EventLoop::new();

        if let Some(fullscreen) = builder.fullscreen {
            window_builder = window_builder.with_fullscreen(match fullscreen {
                FullscreenMode::Borderless => Some(Fullscreen::Borderless(None)),
                FullscreenMode::Exclusive => Some(Fullscreen::Exclusive(
                    event_loop
                        .primary_monitor()
                        .expect("at least one monitor")
                        .video_modes()
                        .next()
                        .expect("at least one video mode"),
                )),
            });
        }

        let window = window_builder.build(&event_loop).expect("window");

        // Physical window extent in pixels
        let swapchain_extent = [window.inner_size().width, window.inner_size().height];

        // Find the internal rendering resolution
        let render_extent = [
            (builder.resolution[0] as f32 / builder.temporal_upsampling) as u32,
            (builder.resolution[1] as f32 / builder.temporal_upsampling) as u32,
        ];

        log::info!(
            "Internal rendering extent: {}x{}",
            render_extent[0],
            render_extent[1]
        );

        let temporal_upscale_extent = builder.resolution;

        if builder.temporal_upsampling != 1.0 {
            log::info!(
                "Temporal upscaling extent: {}x{}",
                temporal_upscale_extent[0],
                temporal_upscale_extent[1]
            );
        }

        let render_backend = RenderBackend::new(
            &window,
            RenderBackendConfig {
                swapchain_extent,
                vsync: builder.vsync,
                graphics_debugging: builder.graphics_debugging,
            },
        )?;

        let lazy_cache = LazyCache::create();
        let world_renderer = WorldRenderer::new(
            render_extent,
            temporal_upscale_extent,
            &render_backend,
            &lazy_cache,
        )?;
        let ui_renderer = UiRenderer::default();

        let rg_renderer = kajiya::rg::renderer::Renderer::new(&render_backend)?;

        #[cfg(feature = "dear-imgui")]
        let mut imgui = imgui::Context::create();

        #[cfg(feature = "dear-imgui")]
        let mut imgui_backend =
            kajiya_imgui::ImGuiBackend::new(rg_renderer.device().clone(), &window, &mut imgui);

        #[cfg(feature = "dear-imgui")]
        imgui_backend.create_graphics_resources(swapchain_extent);

        #[cfg(feature = "use-egui")]
        let mut egui = egui::Context::default();

        #[cfg(feature = "use-egui")]
        {
            egui.set_visuals(EguiContext::get_theme_visuals());
            let mut style: egui::Style = (*egui.style()).clone();
            style.override_text_style = Some(TextStyle::Monospace);
            egui.set_style(style);
        }

        #[cfg(feature = "use-egui")]
        let (window_size, window_scale_factor) = (
            (window.inner_size().width, window.inner_size().height),
            window.scale_factor(),
        );

        #[cfg(feature = "use-egui")]
        let mut egui_backend = kajiya_egui::EguiBackend::new(
            rg_renderer.device().clone(),
            window_size,
            window_scale_factor,
            &mut egui,
        );

        #[cfg(feature = "use-egui")]
        egui_backend.create_graphics_resources([window_size.0, window_size.1]);

        #[cfg(feature = "use-egui")]
        let egui = EguiState {
            egui_context: egui,
            raw_input: egui_backend.raw_input.clone(),
            window_size,
            window_scale_factor,
            last_mouse_pos: None,
            last_dt: 0.0,
        };

        #[cfg(feature = "puffin-server")]
        let puffin_server = {
            let server_addr = format!("0.0.0.0:{}", puffin_http::DEFAULT_PORT);
            log::info!("Serving profile data on {}", server_addr);

            puffin::set_scopes_on(true);
            puffin_http::Server::new(&server_addr).unwrap()
        };

        let optional = MainLoopOptional {
            #[cfg(feature = "dear-imgui")]
            imgui_backend,
            #[cfg(feature = "dear-imgui")]
            imgui,
            #[cfg(feature = "use-egui")]
            egui_backend,
            #[cfg(feature = "use-egui")]
            egui,
            #[cfg(feature = "puffin-server")]
            _puffin_server: puffin_server,
        };

        Ok(Self {
            window,
            world_renderer,
            ui_renderer,
            optional,
            event_loop,
            render_backend,
            rg_renderer,
            render_extent,
        })
    }

    pub fn window_aspect_ratio(&self) -> f32 {
        self.window.inner_size().width as f32 / self.window.inner_size().height as f32
    }

    pub fn run<'a, FrameFn>(self, mut frame_fn: FrameFn) -> anyhow::Result<()>
    where
        FrameFn: (FnMut(FrameContext) -> WorldFrameDesc) + 'a,
    {
        #[allow(unused_variables, unused_mut)]
        let SimpleMainLoop {
            window,
            mut world_renderer,
            mut ui_renderer,
            mut optional,
            mut event_loop,
            mut render_backend,
            mut rg_renderer,
            render_extent,
        } = self;

        let mut events = Vec::new();

        let mut last_frame_instant = std::time::Instant::now();
        let mut last_error_text = None;

        // Delta times are filtered over _this many_ frames.
        const DT_FILTER_WIDTH: usize = 10;

        // Past delta times used for filtering
        let mut dt_queue: VecDeque<f32> = VecDeque::with_capacity(DT_FILTER_WIDTH);

        // Fake the first frame's delta time. In the first frame, shaders
        // and pipelines are be compiled, so it will most likely have a spike.
        let mut fake_dt_countdown: i32 = 1;

        let mut running = true;
        while running {
            let gpu_frame_start_ns = puffin::now_ns();
            puffin::profile_scope!("main loop");
            puffin::GlobalProfiler::lock().new_frame();

            event_loop.run_return(|event, _, control_flow| {
                puffin::profile_scope!("event handler");

                let _ = &render_backend;
                #[cfg(feature = "dear-imgui")]
                optional
                    .imgui_backend
                    .handle_event(&window, &mut optional.imgui, &event);

                #[cfg(feature = "dear-imgui")]
                let ui_wants_mouse = optional.imgui.io().want_capture_mouse;

                #[cfg(not(feature = "dear-imgui"))]
                let ui_wants_mouse = false;

                *control_flow = ControlFlow::Poll;

                match event {
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                            running = false;
                        }
                        WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. }
                            if ui_wants_mouse => {}
                        _ => events.extend(event.to_static()),
                    },
                    Event::MainEventsCleared => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                }
            });

            puffin::profile_scope!("MainEventsCleared");

            // Filter the frame time before passing it to the application and renderer.
            // Fluctuations in frame rendering times cause stutter in animations,
            // and time-dependent effects (such as motion blur).
            //
            // Should applications need unfiltered delta time, they can calculate
            // it themselves, but it's good to pass the filtered time so users
            // don't need to worry about it.
            let dt_filtered = {
                let now = std::time::Instant::now();
                let dt_duration = now - last_frame_instant;
                last_frame_instant = now;

                let dt_raw = dt_duration.as_secs_f32();

                // >= because rendering (and thus the spike) happens _after_ this.
                if fake_dt_countdown >= 0 {
                    // First frame. Return the fake value.
                    fake_dt_countdown -= 1;
                    dt_raw.min(1.0 / 60.0)
                } else {
                    // Not the first frame. Start averaging.

                    if dt_queue.len() >= DT_FILTER_WIDTH {
                        dt_queue.pop_front();
                    }

                    dt_queue.push_back(dt_raw);
                    dt_queue.iter().copied().sum::<f32>() / dt_queue.len() as f32
                }
            };

            let frame_desc = frame_fn(FrameContext {
                dt_filtered,
                render_extent,
                events: &events,
                world_renderer: &mut world_renderer,

                #[cfg(feature = "dear-imgui")]
                imgui: Some(ImguiContext {
                    imgui: &mut optional.imgui,
                    imgui_backend: &mut optional.imgui_backend,
                    ui_renderer: &mut ui_renderer,
                    dt_filtered,
                    window: &window,
                }),

                #[cfg(feature = "use-egui")]
                egui: Some(EguiContext {
                    egui: &mut optional.egui,
                    egui_backend: &mut optional.egui_backend,
                    ui_renderer: &mut ui_renderer,
                    dt_filtered,
                }),
            });

            events.clear();

            // Physical window extent in pixels
            let swapchain_extent = [window.inner_size().width, window.inner_size().height];

            let prepared_frame = {
                puffin::profile_scope!("prepare_frame");
                rg_renderer.prepare_frame(|rg| {
                    rg.debug_hook = world_renderer.rg_debug_hook.take();
                    let main_img = world_renderer.prepare_render_graph(rg, &frame_desc);
                    let ui_img = ui_renderer.prepare_render_graph(rg);

                    let mut swap_chain = rg.get_swap_chain();
                    rg::SimpleRenderPass::new_compute(
                        rg.add_pass("final blit"),
                        "/shaders/final_blit.hlsl",
                    )
                    .read(&main_img)
                    .read(&ui_img)
                    .write(&mut swap_chain)
                    .constants((
                        main_img.desc().extent_inv_extent_2d(),
                        [
                            swapchain_extent[0] as f32,
                            swapchain_extent[1] as f32,
                            1.0 / swapchain_extent[0] as f32,
                            1.0 / swapchain_extent[1] as f32,
                        ],
                    ))
                    .dispatch([swapchain_extent[0], swapchain_extent[1], 1]);
                })
            };

            match prepared_frame {
                Ok(()) => {
                    puffin::profile_scope!("draw_frame");
                    rg_renderer.draw_frame(
                        |dynamic_constants| {
                            world_renderer.prepare_frame_constants(
                                dynamic_constants,
                                &frame_desc,
                                dt_filtered,
                            )
                        },
                        &mut render_backend.swapchain,
                    );
                    world_renderer.retire_frame();
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

            report_gpu_stats_to_puffin(&gpu_profiler::get_stats(), gpu_frame_start_ns);
        }

        Ok(())
    }
}

fn report_gpu_stats_to_puffin(
    gpu_stats: &gpu_profiler::GpuProfilerStats,
    gpu_frame_start_ns: puffin::NanoSecond,
) {
    let mut stream = puffin::Stream::default();
    let gpu_scopes = gpu_stats.get_ordered();
    let mut gpu_time_accum: puffin::NanoSecond = 0;
    let mut puffin_scope_count = 0;
    let main_gpu_scope_offset = stream.begin_scope(gpu_frame_start_ns, "frame", "", "");
    puffin_scope_count += 1;
    puffin_scope_count += gpu_scopes.len();
    for (scope, ms) in gpu_scopes {
        let ns = (ms * 1_000_000.0) as puffin::NanoSecond;
        let offset = stream.begin_scope(gpu_frame_start_ns + gpu_time_accum, &scope.name, "", "");
        gpu_time_accum += ns;
        stream.end_scope(offset, gpu_frame_start_ns + gpu_time_accum);
    }
    stream.end_scope(main_gpu_scope_offset, gpu_frame_start_ns + gpu_time_accum);
    puffin::global_reporter(
        puffin::ThreadInfo {
            start_time_ns: None,
            name: "gpu".to_owned(),
        },
        &puffin::StreamInfo {
            num_scopes: puffin_scope_count,
            stream,
            depth: 1,
            range_ns: (gpu_frame_start_ns, gpu_frame_start_ns + gpu_time_accum),
        }
        .as_stream_into_ref(),
    );
}
