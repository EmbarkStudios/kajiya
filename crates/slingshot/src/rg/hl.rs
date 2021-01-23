use ash::vk;
use vk_sync::AccessType;

use crate::{backend::image::ImageViewDescBuilder, Image};

use super::{
    BindRgRef, Buffer, GpuSrv, GpuUav, Handle, PassBuilder, Ref, RenderPassBinding, Resource,
    RgComputePipelineHandle,
};

pub struct SimpleComputePass<'rg> {
    pass: PassBuilder<'rg>,
    pipeline: RgComputePipelineHandle,
    bindings: Vec<RenderPassBinding>,
}

impl<'rg> SimpleComputePass<'rg> {
    pub fn new(mut pass: PassBuilder<'rg>, pipeline_path: &str) -> Self {
        let pipeline = pass.register_compute_pipeline(pipeline_path);

        Self {
            pass,
            pipeline,
            bindings: Vec::new(),
        }
    }

    pub fn read<Res>(mut self, handle: &Handle<Res>) -> Self
    where
        Res: Resource + 'static,
        Ref<Res, GpuSrv>: BindRgRef,
    {
        let handle_ref = self.pass.read(
            handle,
            AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
        );

        self.bindings.push(BindRgRef::bind(&handle_ref));

        self
    }

    pub fn read_aspect(
        mut self,
        handle: &Handle<Image>,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Self {
        let handle_ref = self.pass.read(
            handle,
            AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
        );

        self.bindings
            .push(handle_ref.bind_view(ImageViewDescBuilder::default().aspect_mask(aspect_mask)));

        self
    }

    pub fn write<Res>(mut self, handle: &mut Handle<Res>) -> Self
    where
        Res: Resource + 'static,
        Ref<Res, GpuUav>: BindRgRef,
    {
        let handle_ref = self.pass.write(handle, AccessType::ComputeShaderWrite);

        self.bindings.push(BindRgRef::bind(&handle_ref));

        self
    }

    pub fn dispatch(self, extent: [u32; 3]) {
        let pipeline = self.pipeline;
        let bindings = self.bindings;

        self.pass.render(move |api| {
            let pipeline =
                api.bind_compute_pipeline(pipeline.into_binding().descriptor_set(0, &bindings));

            pipeline.dispatch(extent);
        });
    }

    pub fn dispatch_indirect(mut self, args_buffer: &Handle<Buffer>, args_buffer_offset: u64) {
        let args_buffer_ref = self.pass.read(args_buffer, AccessType::IndirectBuffer);

        let pipeline = self.pipeline;
        let bindings = self.bindings;

        self.pass.render(move |api| {
            let pipeline =
                api.bind_compute_pipeline(pipeline.into_binding().descriptor_set(0, &bindings));

            pipeline.dispatch_indirect(args_buffer_ref, args_buffer_offset);
        });
    }
}
