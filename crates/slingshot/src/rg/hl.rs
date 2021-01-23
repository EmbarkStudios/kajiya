use ash::vk;
use vk_sync::AccessType;

use crate::{backend::image::ImageViewDescBuilder, Image};

use super::{
    BindRgRef, Buffer, GpuSrv, GpuUav, Handle, PassBuilder, Ref, RenderPassApi, RenderPassBinding,
    Resource, RgComputePipelineHandle,
};

trait ConstBlob {
    fn push_self(
        self: Box<Self>,
        dynamic_constants: &mut crate::dynamic_constants::DynamicConstants,
    ) -> u32;
}

impl<T> ConstBlob for T
where
    T: Copy + 'static,
{
    fn push_self(
        self: Box<Self>,
        dynamic_constants: &mut crate::dynamic_constants::DynamicConstants,
    ) -> u32 {
        dynamic_constants.push(*self)
    }
}

pub struct SimpleComputePassState {
    pipeline: RgComputePipelineHandle,
    bindings: Vec<RenderPassBinding>,
    const_blobs: Vec<(usize, Box<dyn ConstBlob>)>,
    raw_descriptor_sets: Vec<(u32, vk::DescriptorSet)>,
}

impl SimpleComputePassState {
    pub fn new(pipeline: RgComputePipelineHandle) -> Self {
        Self {
            pipeline,
            bindings: Vec::new(),
            const_blobs: Vec::new(),
            raw_descriptor_sets: Vec::new(),
        }
    }

    fn patch_const_blobs(&mut self, api: &mut RenderPassApi) {
        let dynamic_constants = api.dynamic_constants();

        let const_blobs = std::mem::take(&mut self.const_blobs);
        for (binding_idx, blob) in const_blobs {
            let dynamic_constants_offset = ConstBlob::push_self(blob, dynamic_constants);
            match &mut self.bindings[binding_idx] {
                RenderPassBinding::DynamicConstants(offset) => {
                    *offset = dynamic_constants_offset;
                }
                _ => unreachable!(),
            }
        }
    }
}

pub struct SimpleComputePass<'rg> {
    pass: PassBuilder<'rg>,
    state: SimpleComputePassState,
}

impl<'rg> SimpleComputePass<'rg> {
    pub fn new(mut pass: PassBuilder<'rg>, pipeline_path: &str) -> Self {
        let pipeline = pass.register_compute_pipeline(pipeline_path);

        Self {
            pass,
            state: SimpleComputePassState::new(pipeline),
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

        self.state.bindings.push(BindRgRef::bind(&handle_ref));

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

        self.state
            .bindings
            .push(handle_ref.bind_view(ImageViewDescBuilder::default().aspect_mask(aspect_mask)));

        self
    }

    pub fn write<Res>(mut self, handle: &mut Handle<Res>) -> Self
    where
        Res: Resource + 'static,
        Ref<Res, GpuUav>: BindRgRef,
    {
        let handle_ref = self.pass.write(handle, AccessType::ComputeShaderWrite);

        self.state.bindings.push(BindRgRef::bind(&handle_ref));

        self
    }

    pub fn constants<T: Copy + 'static>(mut self, consts: T) -> Self {
        let binding_idx = self.state.bindings.len();

        self.state
            .bindings
            .push(RenderPassBinding::DynamicConstants(0));
        self.state.const_blobs.push((binding_idx, Box::new(consts)));

        self
    }

    pub fn raw_descriptor_set(mut self, set_idx: u32, set: vk::DescriptorSet) -> Self {
        self.state.raw_descriptor_sets.push((set_idx, set));
        self
    }

    pub fn dispatch(self, extent: [u32; 3]) {
        let mut state = self.state;

        self.pass.render(move |api| {
            state.patch_const_blobs(api);

            let pipeline = api.bind_compute_pipeline(
                state
                    .pipeline
                    .into_binding()
                    .descriptor_set(0, &state.bindings),
            );

            pipeline.dispatch(extent);
        });
    }

    pub fn dispatch_indirect(mut self, args_buffer: &Handle<Buffer>, args_buffer_offset: u64) {
        let args_buffer_ref = self.pass.read(args_buffer, AccessType::IndirectBuffer);
        let mut state = self.state;

        self.pass.render(move |api| {
            state.patch_const_blobs(api);

            let pipeline = api.bind_compute_pipeline(
                state
                    .pipeline
                    .into_binding()
                    .descriptor_set(0, &state.bindings),
            );

            pipeline.dispatch_indirect(args_buffer_ref, args_buffer_offset);
        });
    }
}
