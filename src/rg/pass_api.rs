use std::sync::Arc;

use ash::version::DeviceV1_0;

use super::{ResourceRegistry, RgComputePipelineHandle};
use crate::{
    backend::{device::CommandBuffer, shader::ShaderPipeline},
    renderer::bind_descriptor_set,
    renderer::DescriptorSetBinding,
};

pub struct RenderPassApi<'a, 'exec_params, 'constants> {
    pub cb: &'a CommandBuffer,
    pub resources: &'a ResourceRegistry<'exec_params, 'constants>,
}

pub struct RenderPassComputePipelineBinding<'a> {
    pipeline: RgComputePipelineHandle,
    use_frame_constants: bool,

    // TODO: fixed size
    descriptor_set_bindings: Vec<(u32, &'a [DescriptorSetBinding])>,
}

impl<'a> RenderPassComputePipelineBinding<'a> {
    pub fn new(pipeline: RgComputePipelineHandle) -> Self {
        Self {
            pipeline,
            use_frame_constants: false,
            descriptor_set_bindings: Vec::new(),
        }
    }

    pub fn use_frame_constants(mut self, use_frame_constants: bool) -> Self {
        self.use_frame_constants = true;
        self
    }

    pub fn descriptor_set(
        mut self,
        set_idx: u32,
        descriptor_set_bindings: &'a [DescriptorSetBinding],
    ) -> Self {
        self.descriptor_set_bindings
            .push((set_idx, descriptor_set_bindings));
        self
    }
}

impl<'a, 'exec_params, 'constants> RenderPassApi<'a, 'exec_params, 'constants> {
    /*#[inline(always)]
    pub fn get_compute_pipeline(&self, handle: RgComputePipelineHandle) -> Arc<ShaderPipeline> {
        self.resources.compute_pipeline(handle)
    }

    #[inline(always)]
    pub fn bind_pipeline(&mut self, pipeline: &ShaderPipeline) {
        unsafe {
            self.resources
                .execution_params
                .device
                .raw
                .cmd_bind_pipeline(self.cb.raw, pipeline.pipeline_bind_point, pipeline.pipeline);
        }
    }

    #[inline(always)]
    pub fn bind_frame_constants(&self, pipeline: &ShaderPipeline) {
        self.resources.bind_frame_constants(self.cb, pipeline)
    }

    #[inline(always)]
    pub fn bind_descriptor_set(
        &mut self,
        pipeline: &ShaderPipeline,
        set_index: u32,
        bindings: &[DescriptorSetBinding],
    ) {
        bind_descriptor_set(
            &*self.resources.execution_params.device,
            self.cb,
            pipeline,
            set_index,
            bindings,
        );
    }*/

    pub fn bind_pipeline(&mut self, binding: RenderPassComputePipelineBinding<'_>) {
        let device = self.resources.execution_params.device;
        let pipeline = self.resources.compute_pipeline(binding.pipeline);
        let pipeline = &*pipeline;

        unsafe {
            device.raw.cmd_bind_pipeline(
                self.cb.raw,
                pipeline.pipeline_bind_point,
                pipeline.pipeline,
            );
        }

        if binding.use_frame_constants {
            self.resources.bind_frame_constants(self.cb, pipeline);
        }

        for (set_index, bindings) in binding.descriptor_set_bindings {
            bind_descriptor_set(
                &*self.resources.execution_params.device,
                self.cb,
                pipeline,
                set_index,
                bindings,
            );
        }
    }
}
