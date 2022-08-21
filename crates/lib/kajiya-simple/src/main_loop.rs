use std::collections::VecDeque;

use kajiya::{
    backend::{vulkan::RenderBackendConfig, *},
    frame_desc::WorldFrameDesc,
    rg,
    ui_renderer::UiRenderer,
    world_renderer::WorldRenderer,
};

#[cfg(feature = "dear-imgui")]
use kajiya_imgui::ImGuiBackend;

use turbosloth::*;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::{Fullscreen, WindowBuilder},
};

pub struct FrameContext<'a> {
    pub dt_filtered: f32,
    pub render_extent: [u32; 2],
    pub events: &'a [Event<'static, ()>],
    pub world_renderer: &'a mut WorldRenderer,
    pub window: &'a winit::window::Window,

    #[cfg(feature = "dear-imgui")]
    pub imgui: Option<ImguiContext<'a>>,
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
    physical_device_index: Option<usize>,
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
            physical_device_index: None,
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

    pub fn physical_device_index(mut self, physical_device_index: Option<usize>) -> Self {
        self.physical_device_index = physical_device_index;
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
                device_index: builder.physical_device_index,
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
            gpu_profiler::profiler().begin_frame();
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

                let mut allow_event = true;
                match &event {
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                            running = false;
                        }
                        WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. }
                            if ui_wants_mouse =>
                        {
                            allow_event = false;
                        }
                        _ => {}
                    },
                    Event::MainEventsCleared => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                }

                if allow_event {
                    events.extend(event.to_static());
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
                window: &window,

                #[cfg(feature = "dear-imgui")]
                imgui: Some(ImguiContext {
                    imgui: &mut optional.imgui,
                    imgui_backend: &mut optional.imgui_backend,
                    ui_renderer: &mut ui_renderer,
                    dt_filtered,
                    window: &window,
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

            gpu_profiler::profiler().end_frame();
            if let Some(report) = gpu_profiler::profiler().last_report() {
                report.send_to_puffin(gpu_frame_start_ns);
            };
        }

        Ok(())
    }
}
