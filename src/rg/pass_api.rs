use std::sync::Arc;

use arrayvec::ArrayVec;
use ash::{version::DeviceV1_0, vk};

use super::{
    Buffer, GpuRt, GpuSrv, GpuUav, GraphRawResourceHandle, Image, Ref, ResourceRegistry,
    RgComputePipelineHandle, RgRasterPipelineHandle,
};
use crate::{
    backend::shader::FramebufferCacheKey,
    backend::shader::MAX_COLOR_ATTACHMENTS,
    backend::{
        device::{CommandBuffer, Device},
        image::{ImageViewDesc, ImageViewDescBuilder},
        shader::{ComputePipeline, RasterPipeline},
    },
    renderer::{bind_descriptor_set, DescriptorSetBinding},
};

pub struct RenderPassApi<'a, 'exec_params, 'constants> {
    pub cb: &'a CommandBuffer,
    pub resources: &'a ResourceRegistry<'exec_params, 'constants>,
}

pub struct RenderPassComputePipelineBinding<'a> {
    pipeline: RgComputePipelineHandle,

    // TODO: fixed size
    bindings: Vec<(u32, &'a [RenderPassBinding])>,
}

impl<'a> RenderPassComputePipelineBinding<'a> {
    pub fn new(pipeline: RgComputePipelineHandle) -> Self {
        Self {
            pipeline,
            bindings: Vec::new(),
        }
    }

    pub fn descriptor_set(mut self, set_idx: u32, bindings: &'a [RenderPassBinding]) -> Self {
        self.bindings.push((set_idx, bindings));
        self
    }
}

impl RgComputePipelineHandle {
    pub fn into_binding<'a>(self) -> RenderPassComputePipelineBinding<'a> {
        RenderPassComputePipelineBinding::new(self)
    }
}

pub struct RenderPassRasterPipelineBinding<'a> {
    pipeline: RgRasterPipelineHandle,

    // TODO: fixed size
    bindings: Vec<(u32, &'a [RenderPassBinding])>,
}

impl<'a> RenderPassRasterPipelineBinding<'a> {
    pub fn new(pipeline: RgRasterPipelineHandle) -> Self {
        Self {
            pipeline,
            bindings: Vec::new(),
        }
    }

    pub fn descriptor_set(mut self, set_idx: u32, bindings: &'a [RenderPassBinding]) -> Self {
        self.bindings.push((set_idx, bindings));
        self
    }
}

impl RgRasterPipelineHandle {
    pub fn into_binding<'a>(self) -> RenderPassRasterPipelineBinding<'a> {
        RenderPassRasterPipelineBinding::new(self)
    }
}

impl<'a, 'exec_params, 'constants> RenderPassApi<'a, 'exec_params, 'constants> {
    pub fn device(&self) -> &Device {
        self.resources.execution_params.device
    }

    pub fn bind_compute_pipeline<'s>(
        &'s mut self,
        binding: RenderPassComputePipelineBinding<'_>,
    ) -> BoundComputePipeline<'s, 'a, 'exec_params, 'constants> {
        let device = self.resources.execution_params.device;
        let pipeline_arc = self.resources.compute_pipeline(binding.pipeline);
        let pipeline = &*pipeline_arc;

        unsafe {
            device.raw.cmd_bind_pipeline(
                self.cb.raw,
                pipeline.pipeline_bind_point,
                pipeline.pipeline,
            );
        }

        if pipeline
            .set_layout_info
            .get(2)
            .map(|set| !set.is_empty())
            .unwrap_or_default()
        {
            unsafe {
                device.raw.cmd_bind_descriptor_sets(
                    self.cb.raw,
                    pipeline.pipeline_bind_point,
                    pipeline.pipeline_layout,
                    2,
                    &[self.resources.execution_params.frame_descriptor_set],
                    &[self.resources.execution_params.frame_constants_offset],
                );
            }
        }

        for (set_index, bindings) in binding.bindings {
            let bindings = bindings
                .iter()
                .map(|binding| match binding {
                    RenderPassBinding::Image(image) => DescriptorSetBinding::Image(
                        vk::DescriptorImageInfo::builder()
                            .image_layout(image.image_layout)
                            .image_view(self.resources.image_view(image.handle, &image.view_desc))
                            .build(),
                    ),
                    RenderPassBinding::Buffer(buffer) => DescriptorSetBinding::Buffer(
                        vk::DescriptorBufferInfo::builder()
                            .buffer(
                                self.resources
                                    .buffer_from_raw_handle::<GpuSrv>(buffer.handle)
                                    .raw,
                            )
                            .range(vk::WHOLE_SIZE)
                            .build(),
                    ),
                })
                .collect::<Vec<_>>();

            bind_descriptor_set(
                &*self.resources.execution_params.device,
                self.cb,
                pipeline,
                set_index,
                &bindings,
            );
        }

        BoundComputePipeline {
            api: self,
            pipeline: pipeline_arc,
        }
    }

    pub fn bind_raster_pipeline<'s>(
        &'s mut self,
        binding: RenderPassRasterPipelineBinding<'_>,
    ) -> BoundRasterPipeline<'s, 'a, 'exec_params, 'constants> {
        let device = self.resources.execution_params.device;
        let pipeline_arc = self.resources.raster_pipeline(binding.pipeline);
        let pipeline = &*pipeline_arc;

        unsafe {
            device.raw.cmd_bind_pipeline(
                self.cb.raw,
                pipeline.pipeline_bind_point,
                pipeline.pipeline,
            );
        }

        if pipeline
            .set_layout_info
            .get(2)
            .map(|set| !set.is_empty())
            .unwrap_or_default()
        {
            unsafe {
                device.raw.cmd_bind_descriptor_sets(
                    self.cb.raw,
                    pipeline.pipeline_bind_point,
                    pipeline.pipeline_layout,
                    2,
                    &[self.resources.execution_params.frame_descriptor_set],
                    &[self.resources.execution_params.frame_constants_offset],
                );
            }
        }

        for (set_index, bindings) in binding.bindings {
            let bindings = bindings
                .iter()
                .map(|binding| match binding {
                    RenderPassBinding::Image(image) => DescriptorSetBinding::Image(
                        vk::DescriptorImageInfo::builder()
                            .image_layout(image.image_layout)
                            .image_view(self.resources.image_view(image.handle, &image.view_desc))
                            .build(),
                    ),
                    RenderPassBinding::Buffer(buffer) => DescriptorSetBinding::Buffer(
                        vk::DescriptorBufferInfo::builder()
                            .buffer(
                                self.resources
                                    .buffer_from_raw_handle::<GpuSrv>(buffer.handle)
                                    .raw,
                            )
                            .range(vk::WHOLE_SIZE)
                            .build(),
                    ),
                })
                .collect::<Vec<_>>();

            bind_descriptor_set(
                &*self.resources.execution_params.device,
                self.cb,
                pipeline,
                set_index,
                &bindings,
            );
        }

        BoundRasterPipeline {
            api: self,
            pipeline: pipeline_arc,
        }
    }

    pub fn begin_render_pass(
        &mut self,
        render_pass: &crate::backend::shader::RenderPass,
        dims: [u32; 2],
        color_attachments: &[(Ref<Image, GpuRt>, &ImageViewDesc)],
        depth_attachment: Option<(Ref<Image, GpuRt>, &ImageViewDesc)>,
    ) {
        let device = self.resources.execution_params.device;

        let framebuffer = render_pass
            .framebuffer_cache
            .get_or_create(
                &device.raw,
                FramebufferCacheKey::new(
                    dims,
                    color_attachments.iter().map(|(a, _)| {
                        &self.resources.image_from_raw_handle::<GpuRt>(a.handle).desc
                    }),
                    depth_attachment.as_ref().map(|(a, _)| {
                        &self.resources.image_from_raw_handle::<GpuRt>(a.handle).desc
                    }),
                ),
            )
            .unwrap();

        // Bind images to the imageless framebuffer
        let image_attachments: ArrayVec<[vk::ImageView; MAX_COLOR_ATTACHMENTS + 1]> =
            color_attachments
                .iter()
                .chain(depth_attachment.as_ref().into_iter())
                .map(|(img, view)| self.resources.image_view(img.handle, view.clone()))
                .collect();

        let mut pass_attachment_desc =
            vk::RenderPassAttachmentBeginInfoKHR::builder().attachments(&image_attachments);

        let [width, height] = dims;

        //.clear_values(&clear_values)
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
            .push_next(&mut pass_attachment_desc);

        unsafe {
            device.raw.cmd_begin_render_pass(
                self.cb.raw,
                &pass_begin_desc,
                vk::SubpassContents::INLINE,
            );
        }
    }

    pub fn end_render_pass(&mut self) {
        let device = self.resources.execution_params.device;
        unsafe {
            device.raw.cmd_end_render_pass(self.cb.raw);
        }
    }

    pub fn set_default_view_and_scissor(&mut self, [width, height]: [u32; 2]) {
        let raw_device = &self.resources.execution_params.device.raw;
        let cb_raw = self.cb.raw;

        unsafe {
            raw_device.cmd_set_viewport(
                cb_raw,
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

            raw_device.cmd_set_scissor(
                cb_raw,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: width as _,
                        height: height as _,
                    },
                }],
            );
        }
    }
}

pub struct BoundComputePipeline<'api, 'a, 'exec_params, 'constants> {
    api: &'api mut RenderPassApi<'a, 'exec_params, 'constants>,
    pipeline: Arc<ComputePipeline>,
}

impl<'api, 'a, 'exec_params, 'constants> BoundComputePipeline<'api, 'a, 'exec_params, 'constants> {
    pub fn dispatch(&self, threads: [u32; 3]) {
        let group_size = self.pipeline.group_size;

        unsafe {
            self.api.device().raw.cmd_dispatch(
                self.api.cb.raw,
                (threads[0] + group_size[0] - 1) / group_size[0],
                (threads[1] + group_size[1] - 1) / group_size[1],
                (threads[2] + group_size[2] - 1) / group_size[2],
            );
        }
    }
}

pub struct BoundRasterPipeline<'api, 'a, 'exec_params, 'constants> {
    api: &'api mut RenderPassApi<'a, 'exec_params, 'constants>,
    pipeline: Arc<RasterPipeline>,
}

pub struct RenderPassImageBinding {
    handle: GraphRawResourceHandle,
    view_desc: ImageViewDesc,
    image_layout: vk::ImageLayout,
}

pub struct RenderPassBufferBinding {
    handle: GraphRawResourceHandle,
}

pub enum RenderPassBinding {
    Image(RenderPassImageBinding),
    Buffer(RenderPassBufferBinding),
}

impl Ref<Image, GpuSrv> {
    pub fn bind(&self, view_desc: ImageViewDescBuilder) -> RenderPassBinding {
        RenderPassBinding::Image(RenderPassImageBinding {
            handle: self.handle,
            view_desc: view_desc.build().unwrap(),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        })
    }
}

impl Ref<Image, GpuUav> {
    pub fn bind(&self, view_desc: ImageViewDescBuilder) -> RenderPassBinding {
        RenderPassBinding::Image(RenderPassImageBinding {
            handle: self.handle,
            view_desc: view_desc.build().unwrap(),
            image_layout: vk::ImageLayout::GENERAL,
        })
    }
}

impl Ref<Buffer, GpuSrv> {
    pub fn bind(&self) -> RenderPassBinding {
        RenderPassBinding::Buffer(RenderPassBufferBinding {
            handle: self.handle,
        })
    }
}

impl Ref<Buffer, GpuUav> {
    pub fn bind(&self) -> RenderPassBinding {
        RenderPassBinding::Buffer(RenderPassBufferBinding {
            handle: self.handle,
        })
    }
}
