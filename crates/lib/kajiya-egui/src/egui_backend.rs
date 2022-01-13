use std::sync::Arc;

use ash_egui::egui::{self, vec2, CtxRef};
use kajiya::{
    backend::{
        ash::{self, vk},
        Device, Image, ImageDesc, ImageViewDesc,
    },
    ui_renderer::UiRenderer,
};

use parking_lot::Mutex;

struct GfxResources {
    //egui_render_pass: RenderPass,
    pub egui_render_pass: vk::RenderPass,
    pub egui_framebuffer: vk::Framebuffer,
    pub egui_texture: Arc<Image>,
}

pub struct EguiBackendInner {
    egui_renderer: ash_egui::Renderer,
    gfx: Option<GfxResources>,
}

pub struct EguiBackend {
    inner: Arc<Mutex<EguiBackendInner>>,
    device: Arc<Device>,
    pub raw_input: ash_egui::egui::RawInput,
}

impl EguiBackend {
    pub fn new(
        device: Arc<Device>,
        window_settings: (u32, u32, f64),
        context: &mut CtxRef,
    ) -> Self {
        let (window_width, window_height, window_scale_factor) = window_settings;

        // Create raw_input
        let raw_input = egui::RawInput {
            pixels_per_point: Some(window_scale_factor as f32),
            screen_rect: Some(egui::Rect::from_min_size(
                Default::default(),
                vec2(window_width as f32, window_height as f32) / window_scale_factor as f32,
            )),
            time: Some(0.0),
            ..Default::default()
        };

        let egui_renderer = {
            ash_egui::Renderer::new(
                window_width,
                window_height,
                window_scale_factor,
                &device.raw,
                &device.physical_device().properties,
                &device.physical_device().memory_properties,
                context,
                raw_input.clone(),
            )
        };

        Self {
            device,
            inner: Arc::new(Mutex::new(EguiBackendInner {
                egui_renderer,
                gfx: None,
            })),
            raw_input,
        }
    }

    pub fn create_graphics_resources(&mut self, surface_resolution: [u32; 2]) {
        self.inner
            .lock()
            .create_graphics_resources(self.device.as_ref(), surface_resolution);
    }

    #[allow(dead_code)]
    pub fn destroy_graphics_resources(&mut self) {
        let device = &self.device.raw;

        log::trace!("device_wait_idle");
        unsafe { device.device_wait_idle() }.unwrap();

        let mut inner = self.inner.lock();

        if inner.egui_renderer.has_pipeline() {
            inner.egui_renderer.destroy_pipeline(device);
        }

        if let Some(gfx) = inner.gfx.take() {
            unsafe {
                // TODO
                //device.destroy_render_pass(gfx.egui_render_pass, None);
                device.destroy_framebuffer(gfx.egui_framebuffer, None);
            }
        }
    }

    pub fn handle_event(
        &mut self,
        _window: &winit::window::Window,
        _egui: &mut ash_egui::egui::Context,
        _event: &winit::event::Event<'_, ()>,
    ) {
    }

    pub fn prepare_frame(&mut self, context: &mut CtxRef, dt: f32) {
        // update time
        if let Some(time) = self.raw_input.time {
            self.raw_input.time = Some(time + dt as f64);
        } else {
            self.raw_input.time = Some(0.0);
        }

        context.begin_frame(self.raw_input.take());
    }

    pub fn finish_frame(
        &mut self,
        context: &mut CtxRef,
        gui_render_extent: (u32, u32),
        ui_renderer: &mut UiRenderer,
    ) {
        let ui_target_image = self.inner.lock().get_target_image().unwrap();

        let inner = self.inner.clone();
        let device = self.device.clone();

        let (_, clipped_shapes) = context.end_frame();
        let clipped_meshes = context.tessellate(clipped_shapes);

        ui_renderer.ui_frame = Some((
            Box::new(move |cb| {
                inner
                    .lock()
                    .render(
                        [gui_render_extent.0, gui_render_extent.1],
                        clipped_meshes,
                        device,
                        cb,
                    )
                    .expect("ui.render");
            }),
            ui_target_image,
        ));
    }
}

impl EguiBackendInner {
    fn create_graphics_resources(&mut self, device: &Device, surface_resolution: [u32; 2]) {
        assert!(self.gfx.is_none());

        let egui_render_pass = create_egui_render_pass(&device.raw);
        let (egui_framebuffer, egui_texture) =
            create_egui_framebuffer(device, egui_render_pass, surface_resolution);

        let gfx = GfxResources {
            egui_render_pass,
            egui_framebuffer,
            egui_texture,
        };

        self.egui_renderer
            .create_pipeline(&device.raw, gfx.egui_render_pass);

        self.gfx = Some(gfx);
    }

    fn get_target_image(&self) -> Option<Arc<Image>> {
        self.gfx.as_ref().map(|res| res.egui_texture.clone())
    }

    fn render(
        &mut self,
        physical_size: [u32; 2],
        draw_data: Vec<ash_egui::egui::ClippedMesh>,
        device: Arc<Device>,
        cb: vk::CommandBuffer,
    ) -> Option<Arc<Image>> {
        let device = &device.raw;

        match self.gfx {
            Some(ref gfx) => {
                self.egui_renderer.begin_frame(device, cb);

                {
                    let clear_values = [vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    }];

                    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(gfx.egui_render_pass)
                        .framebuffer(gfx.egui_framebuffer)
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: physical_size[0],
                                height: physical_size[1],
                            },
                        })
                        .clear_values(&clear_values);

                    unsafe {
                        device.cmd_begin_render_pass(
                            cb,
                            &render_pass_begin_info,
                            vk::SubpassContents::INLINE,
                        );
                    }
                }

                self.egui_renderer.render(draw_data, device, cb);

                unsafe {
                    device.cmd_end_render_pass(cb);
                }

                Some(gfx.egui_texture.clone())
            }
            None => None,
        }
    }
}

fn create_egui_render_pass(device: &ash::Device) -> vk::RenderPass {
    let renderpass_attachments = [vk::AttachmentDescription {
        format: vk::Format::R8G8B8A8_UNORM,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ..Default::default()
    }];
    let color_attachment_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let dependencies = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ..Default::default()
    }];

    let subpasses = [vk::SubpassDescription::builder()
        .color_attachments(&color_attachment_refs)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build()];

    let renderpass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&renderpass_attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe {
        device
            .create_render_pass(&renderpass_create_info, None)
            .unwrap()
    }
}

fn create_egui_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    surface_resolution: [u32; 2],
) -> (vk::Framebuffer, Arc<Image>) {
    let tex = device
        .create_image(
            ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, surface_resolution)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT),
            vec![],
        )
        .unwrap();

    let framebuffer_attachments = [tex.view(device, &ImageViewDesc::default())];
    let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
        .render_pass(render_pass)
        .attachments(&framebuffer_attachments)
        .width(surface_resolution[0] as _)
        .height(surface_resolution[1] as _)
        .layers(1);

    let fb = unsafe {
        device
            .raw
            .create_framebuffer(&frame_buffer_create_info, None)
    }
    .expect("create_framebuffer");

    (fb, Arc::new(tex))
}
