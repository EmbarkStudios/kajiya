use crate::backend::{
    self, barrier::*, image::*, presentation::blit_image_to_swapchain, shader::*, RenderBackend,
};

use crate::shader_compiler::CompileShader;
use ash::{version::DeviceV1_0, vk};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;
use turbosloth::*;

pub struct Renderer {
    present_shader: ComputePipeline,
    output_img: Arc<Image>,
    output_img_view: ImageView,

    gradients_shader: ComputePipeline,
    raster_simple_render_pass: Arc<RenderPass>,
    raster_simple: RasterPipeline,
}

pub struct LocalStateTracker<'device> {
    resource: vk::Image,
    current_state: vk_sync::AccessType,
    cb: vk::CommandBuffer,
    device: &'device ash::Device,
}

impl<'device> LocalStateTracker<'device> {
    pub fn new(
        resource: vk::Image,
        current_state: vk_sync::AccessType,
        cb: vk::CommandBuffer,
        device: &'device ash::Device,
    ) -> Self {
        Self {
            resource,
            current_state,
            cb,
            device,
        }
    }

    pub fn barrier(&mut self, state: vk_sync::AccessType) {
        record_image_barrier(
            &self.device,
            self.cb,
            ImageBarrier::new(self.resource, self.current_state, state),
        );

        self.current_state = state;
    }
}

impl Renderer {
    pub fn new(backend: &RenderBackend) -> anyhow::Result<Self> {
        let present_shader = backend::presentation::create_present_compute_shader(&*backend.device);

        let lazy_cache = LazyCache::create();

        let output_img = backend.device.create_image(
            ImageDesc::new_2d([1280, 720])
                .format(vk::Format::R16G16B16A16_SFLOAT)
                //.format(vk::Format::R8G8B8A8_UNORM)
                .usage(
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                )
                .build()
                .unwrap(),
            None,
        )?;

        let output_img_view = backend.device.create_image_view(
            ImageViewDesc::builder()
                .image(output_img.clone())
                .build()
                .unwrap(),
        )?;

        let gradients_shader = smol::block_on(
            CompileShader {
                path: "/assets/shaders/gradients.hlsl".into(),
                profile: "cs".to_owned(),
            }
            .into_lazy()
            .eval(&lazy_cache),
        )?;

        let gradients_shader = create_compute_pipeline(
            &*backend.device,
            ComputePipelineDesc::builder()
                .spirv(&gradients_shader.spirv)
                .entry_name("main")
                .descriptor_set_layout_flags(&[(
                    0,
                    vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
                )])
                .build()
                .unwrap(),
        );

        let vertex_shader = smol::block_on(
            CompileShader {
                path: "/assets/shaders/raster_simple_vs.hlsl".into(),
                profile: "vs".to_owned(),
            }
            .into_lazy()
            .eval(&lazy_cache),
        )?;

        let pixel_shader = smol::block_on(
            CompileShader {
                path: "/assets/shaders/raster_simple_ps.hlsl".into(),
                profile: "ps".to_owned(),
            }
            .into_lazy()
            .eval(&lazy_cache),
        )?;

        let raster_simple_render_pass = create_render_pass(
            &*backend.device,
            RenderPassDesc {
                color_attachments: &[RenderPassAttachmentDesc::new(
                    vk::Format::R16G16B16A16_SFLOAT,
                )
                .garbage_input()],
                depth_attachment: None,
            },
        )?;

        let raster_simple = create_raster_pipeline(
            &*backend.device,
            RasterPipelineDesc {
                shaders: &[
                    RasterShaderDesc::new(RasterStage::Vertex, &vertex_shader.spirv, "main")
                        .build()
                        .unwrap(),
                    RasterShaderDesc::new(RasterStage::Pixel, &pixel_shader.spirv, "main")
                        .build()
                        .unwrap(),
                ],
                render_pass: raster_simple_render_pass.clone(),
            },
        )?;

        Ok(Renderer {
            present_shader,
            output_img,
            output_img_view,
            gradients_shader,
            raster_simple_render_pass,
            raster_simple,
        })
    }

    pub fn draw_frame(&mut self, backend: &mut RenderBackend) {
        // Note: this can be done at the end of the frame, not at the start.
        // The image can be acquired just in time for a blit into it,
        // after all the other rendering commands have been recorded.
        let swapchain_image = backend
            .swapchain
            .acquire_next_image()
            .ok()
            .expect("swapchain image");

        let current_frame = backend.device.current_frame();
        let cb = &current_frame.command_buffer;

        let mut output_img_tracker = LocalStateTracker::new(
            self.output_img.raw,
            vk_sync::AccessType::Nothing,
            cb.raw,
            &backend.device.raw,
        );

        unsafe {
            backend
                .device
                .raw
                .begin_command_buffer(
                    cb.raw,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            output_img_tracker.barrier(vk_sync::AccessType::ComputeShaderWrite);

            let image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(self.output_img_view.raw)
                .build();

            backend.device.raw.cmd_bind_pipeline(
                cb.raw,
                vk::PipelineBindPoint::COMPUTE,
                self.gradients_shader.pipeline,
            );

            // TODO: vkCmdPushDescriptorSetWithTemplateKHR
            backend
                .device
                .cmd_ext
                .push_descriptor
                .cmd_push_descriptor_set(
                    cb.raw,
                    vk::PipelineBindPoint::COMPUTE,
                    self.gradients_shader.pipeline_layout,
                    0,
                    &[vk::WriteDescriptorSet::builder()
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(&image_info))
                        .build()],
                );

            // TODO
            let output_size_pixels = (1280u32, 720u32); // TODO
            backend.device.raw.cmd_dispatch(
                cb.raw,
                (output_size_pixels.0 + 7) / 8,
                (output_size_pixels.1 + 7) / 8,
                1,
            );

            {
                // TODO
                let width = 1280;
                let height = 720;

                output_img_tracker.barrier(vk_sync::AccessType::ColorAttachmentWrite);
                let render_pass = &self.raster_simple_render_pass;
                let raster_pipe = &self.raster_simple;

                let framebuffer = render_pass
                    .framebuffer_cache
                    .get_or_create(
                        &backend.device.raw,
                        FramebufferCacheKey::new([width, height], &[&self.output_img.desc], false),
                    )
                    .unwrap();

                // Bind images to the imageless framebuffer
                let image_attachments = [self.output_img_view.raw];
                let mut pass_attachment_desc =
                    vk::RenderPassAttachmentBeginInfoKHR::builder().attachments(&image_attachments);

                let pass_begin_desc = vk::RenderPassBeginInfo::builder()
                    .render_pass(render_pass.raw)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: width as _,
                            height: height as _,
                        },
                    })
                    .push_next(&mut pass_attachment_desc)
                    //.clear_values(&clear_values)
                    ;

                backend.device.raw.cmd_begin_render_pass(
                    cb.raw,
                    &pass_begin_desc,
                    vk::SubpassContents::INLINE,
                );

                {
                    backend.device.raw.cmd_bind_pipeline(
                        cb.raw,
                        vk::PipelineBindPoint::GRAPHICS,
                        raster_pipe.pipeline,
                    );

                    backend.device.raw.cmd_set_viewport(
                        cb.raw,
                        0,
                        &[vk::Viewport {
                            x: 0.0,
                            y: (height as f32),
                            width: width as _,
                            height: -(height as f32),
                            min_depth: 0.0,
                            max_depth: 1.0,
                        }],
                    );
                    backend.device.raw.cmd_set_scissor(
                        cb.raw,
                        0,
                        &[vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: width as _,
                                height: height as _,
                            },
                        }],
                    );

                    backend.device.raw.cmd_draw(cb.raw, 3, 1, 0, 0);
                }

                backend.device.raw.cmd_end_render_pass(cb.raw);
            }

            output_img_tracker
                .barrier(vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer);

            blit_image_to_swapchain(
                &*backend.device,
                cb,
                &swapchain_image,
                &self.output_img_view,
                &self.present_shader,
            );

            backend.device.raw.end_command_buffer(cb.raw).unwrap();
        }

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(std::slice::from_ref(&swapchain_image.acquire_semaphore))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(std::slice::from_ref(&cb.raw));

        unsafe {
            backend
                .device
                .raw
                .reset_fences(std::slice::from_ref(&cb.submit_done_fence))
                .expect("reset_fences");

            backend
                .device
                .raw
                .queue_submit(
                    backend.device.universal_queue.raw,
                    &[submit_info.build()],
                    cb.submit_done_fence,
                )
                .expect("queue submit failed.");
        }

        backend.swapchain.present_image(swapchain_image, &[]);
        backend.device.finish_frame(current_frame);
    }
}
