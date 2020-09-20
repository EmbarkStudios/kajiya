use std::sync::Arc;

use ash::{version::DeviceV1_0, vk};

use super::{
    GpuUav, GraphRawResourceHandle, Image, Ref, ResourceRegistry, RgComputePipelineHandle,
};
use crate::{
    backend::device::Device,
    backend::image::ImageViewDesc,
    backend::image::ImageViewDescBuilder,
    backend::{device::CommandBuffer, shader::ShaderPipeline},
    renderer::bind_descriptor_set,
    renderer::view,
    renderer::DescriptorSetBinding,
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

        if binding.pipeline.use_frame_constants {
            self.resources.bind_frame_constants(self.cb, pipeline);
        }

        for (set_index, bindings) in binding.bindings {
            let bindings = bindings
                .iter()
                .map(|binding| match binding {
                    RenderPassBinding::Image(image) => DescriptorSetBinding::Image(
                        vk::DescriptorImageInfo::builder()
                            .image_layout(image.image_layout)
                            .image_view(
                                self.resources
                                    .image_view(image.handle, &image.view_desc)
                                    .raw,
                            )
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
}

pub struct BoundComputePipeline<'api, 'a, 'exec_params, 'constants> {
    api: &'api mut RenderPassApi<'a, 'exec_params, 'constants>,
    pipeline: Arc<ShaderPipeline>,
}

impl<'api, 'a, 'exec_params, 'constants> BoundComputePipeline<'api, 'a, 'exec_params, 'constants> {
    pub fn dispatch(&self, threads: [u32; 3]) {
        // TODO
        let group_size = [8, 8, 1];

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

pub struct RenderPassImageBinding {
    handle: GraphRawResourceHandle,
    view_desc: ImageViewDesc,
    image_layout: vk::ImageLayout,
}

pub enum RenderPassBinding {
    Image(RenderPassImageBinding),
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
