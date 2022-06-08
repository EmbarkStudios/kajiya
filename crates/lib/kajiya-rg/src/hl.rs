use kajiya_backend::{
    ash::vk,
    dynamic_constants,
    vk_sync::AccessType,
    vulkan::{
        image::*,
        ray_tracing::{RayTracingAcceleration, RayTracingPipelineDesc},
        shader::{ComputePipelineDesc, PipelineShaderDesc, ShaderPipelineStage, ShaderSource},
    },
};

use crate::Image;

use super::{
    BindRgRef, Buffer, GpuSrv, GpuUav, Handle, PassBuilder, Ref, RenderPassApi, RenderPassBinding,
    Resource, RgComputePipelineHandle, RgRtPipelineHandle,
};

pub trait ConstBlob {
    fn push_self(
        self: Box<Self>,
        dynamic_constants: &mut dynamic_constants::DynamicConstants,
    ) -> u32;
}

impl<T> ConstBlob for T
where
    T: Copy + 'static,
{
    fn push_self(
        self: Box<Self>,
        dynamic_constants: &mut dynamic_constants::DynamicConstants,
    ) -> u32 {
        dynamic_constants.push(self.as_ref())
    }
}

struct VecBlob<T>(Vec<T>);

impl<T> ConstBlob for VecBlob<T>
where
    T: Copy + 'static,
{
    fn push_self(
        self: Box<Self>,
        dynamic_constants: &mut dynamic_constants::DynamicConstants,
    ) -> u32 {
        dynamic_constants.push_from_iter(self.0.into_iter())
    }
}

pub struct SimpleRenderPassState<RgPipelineHandle> {
    pipeline: RgPipelineHandle,
    bindings: Vec<RenderPassBinding>,
    const_blobs: Vec<(usize, Box<dyn ConstBlob>)>,
    raw_descriptor_sets: Vec<(u32, vk::DescriptorSet)>,
}

impl<RgPipelineHandle> SimpleRenderPassState<RgPipelineHandle>
where
    RgPipelineHandle: super::IntoRenderPassPipelineBinding + Copy,
{
    pub fn new(pipeline: RgPipelineHandle) -> Self {
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
                RenderPassBinding::DynamicConstants(offset)
                | RenderPassBinding::DynamicConstantsStorageBuffer(offset) => {
                    *offset = dynamic_constants_offset;
                }
                _ => unreachable!(),
            }
        }
    }

    fn create_pipeline_binding(
        &mut self,
    ) -> super::RenderPassPipelineBinding<'_, RgPipelineHandle> {
        let mut res = self
            .pipeline
            .into_binding()
            .descriptor_set(0, &self.bindings);

        for &(set_idx, binding) in &self.raw_descriptor_sets {
            res = res.raw_descriptor_set(set_idx, binding);
        }

        res
    }
}

pub struct SimpleRenderPass<'rg, RgPipelineHandle> {
    pass: PassBuilder<'rg>,
    state: SimpleRenderPassState<RgPipelineHandle>,
}

impl<'rg> SimpleRenderPass<'rg, RgComputePipelineHandle> {
    pub fn new_compute(mut pass: PassBuilder<'rg>, pipeline_path: &str) -> Self {
        let pipeline = pass.register_compute_pipeline(pipeline_path);

        Self {
            pass,
            state: SimpleRenderPassState::new(pipeline),
        }
    }

    pub fn new_compute_rust(mut pass: PassBuilder<'rg>, entry_name: &str) -> Self {
        let pipeline = pass.register_compute_pipeline_with_desc(
            ComputePipelineDesc::builder()
                .compute_rust(entry_name)
                .build()
                .unwrap(),
        );

        Self {
            pass,
            state: SimpleRenderPassState::new(pipeline),
        }
    }

    pub fn dispatch(self, extent: [u32; 3]) {
        let mut state = self.state;

        self.pass.render(move |api| {
            state.patch_const_blobs(api);

            let pipeline = api.bind_compute_pipeline(state.create_pipeline_binding())?;

            pipeline.dispatch(extent);

            Ok(())
        });
    }

    pub fn dispatch_indirect(mut self, args_buffer: &Handle<Buffer>, args_buffer_offset: u64) {
        let args_buffer_ref = self.pass.read(args_buffer, AccessType::IndirectBuffer);
        let mut state = self.state;

        self.pass.render(move |api| {
            state.patch_const_blobs(api);

            let pipeline = api.bind_compute_pipeline(state.create_pipeline_binding())?;

            pipeline.dispatch_indirect(args_buffer_ref, args_buffer_offset);

            Ok(())
        });
    }
}

impl<'rg> SimpleRenderPass<'rg, RgRtPipelineHandle> {
    pub fn new_rt(
        mut pass: PassBuilder<'rg>,
        rgen: ShaderSource,
        miss: impl IntoIterator<Item = ShaderSource>,
        hit: impl IntoIterator<Item = ShaderSource>,
    ) -> Self {
        let miss = miss.into_iter();
        let hit = hit.into_iter();

        let mut shaders = Vec::with_capacity(1 + miss.size_hint().0 + hit.size_hint().0);

        shaders.push(
            PipelineShaderDesc::builder(ShaderPipelineStage::RayGen)
                .source(rgen)
                .build()
                .unwrap(),
        );
        for source in miss {
            shaders.push(
                PipelineShaderDesc::builder(ShaderPipelineStage::RayMiss)
                    .source(source)
                    .build()
                    .unwrap(),
            );
        }

        for source in hit {
            shaders.push(
                PipelineShaderDesc::builder(ShaderPipelineStage::RayClosestHit)
                    .source(source)
                    .build()
                    .unwrap(),
            );
        }

        let pipeline = pass.register_ray_tracing_pipeline(
            &shaders,
            RayTracingPipelineDesc::default().max_pipeline_ray_recursion_depth(1),
        );

        Self {
            pass,
            state: SimpleRenderPassState::new(pipeline),
        }
    }

    pub fn trace_rays(mut self, tlas: &Handle<RayTracingAcceleration>, extent: [u32; 3]) {
        let tlas_ref = self.pass.read(tlas, AccessType::AnyShaderReadOther);
        let mut state = self.state;

        self.pass.render(move |api| {
            state.patch_const_blobs(api);

            let pipeline = api.bind_ray_tracing_pipeline(
                state
                    .create_pipeline_binding()
                    .descriptor_set(3, &[tlas_ref.bind()]),
            )?;

            pipeline.trace_rays(extent);

            Ok(())
        });
    }

    pub fn trace_rays_indirect(
        mut self,
        tlas: &Handle<RayTracingAcceleration>,
        args_buffer: &Handle<Buffer>,
        args_buffer_offset: u64,
    ) {
        let args_buffer_ref = self.pass.read(args_buffer, AccessType::IndirectBuffer);
        let tlas_ref = self.pass.read(tlas, AccessType::AnyShaderReadOther);
        let mut state = self.state;

        self.pass.render(move |api| {
            state.patch_const_blobs(api);

            let pipeline = api.bind_ray_tracing_pipeline(
                state
                    .create_pipeline_binding()
                    .descriptor_set(3, &[tlas_ref.bind()]),
            )?;

            pipeline.trace_rays_indirect(args_buffer_ref, args_buffer_offset);

            Ok(())
        });
    }
}

impl<'rg, RgPipelineHandle> SimpleRenderPass<'rg, RgPipelineHandle> {
    pub fn read<Res>(mut self, handle: &Handle<Res>) -> Self
    where
        Res: Resource + 'static,
        Ref<Res, GpuSrv>: BindRgRef,
    {
        let handle_ref = self.pass.read(
            handle,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        self.state.bindings.push(BindRgRef::bind(&handle_ref));

        self
    }

    pub fn read_array(mut self, handles: &[Handle<Image>]) -> Self {
        assert!(!handles.is_empty());

        let handle_refs = handles
            .iter()
            .map(|handle| {
                self.pass.read(
                    handle,
                    AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                )
            })
            .collect::<Vec<_>>();

        self.state.bindings.push(BindRgRef::bind(&handle_refs));

        self
    }

    pub fn read_view(mut self, handle: &Handle<Image>, view_desc: ImageViewDescBuilder) -> Self {
        let handle_ref = self.pass.read(
            handle,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        self.state.bindings.push(handle_ref.bind_view(view_desc));

        self
    }

    pub fn read_aspect(
        mut self,
        handle: &Handle<Image>,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Self {
        let handle_ref = self.pass.read(
            handle,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
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
        let handle_ref = self.pass.write(handle, AccessType::AnyShaderWrite);

        self.state.bindings.push(BindRgRef::bind(&handle_ref));

        self
    }

    pub fn write_no_sync<Res>(mut self, handle: &mut Handle<Res>) -> Self
    where
        Res: Resource + 'static,
        Ref<Res, GpuUav>: BindRgRef,
    {
        let handle_ref = self.pass.write_no_sync(handle, AccessType::AnyShaderWrite);

        self.state.bindings.push(BindRgRef::bind(&handle_ref));

        self
    }

    pub fn write_view(
        mut self,
        handle: &mut Handle<Image>,
        view_desc: ImageViewDescBuilder,
    ) -> Self {
        let handle_ref = self.pass.write(handle, AccessType::AnyShaderWrite);

        self.state.bindings.push(handle_ref.bind_view(view_desc));

        self
    }

    pub fn constants<T: ConstBlob + 'static>(mut self, consts: T) -> Self {
        let binding_idx = self.state.bindings.len();

        self.state
            .bindings
            .push(RenderPassBinding::DynamicConstants(0));
        self.state.const_blobs.push((binding_idx, Box::new(consts)));

        self
    }

    pub fn dynamic_storage_buffer<T: ConstBlob + 'static>(mut self, consts: T) -> Self {
        let binding_idx = self.state.bindings.len();

        self.state
            .bindings
            .push(RenderPassBinding::DynamicConstantsStorageBuffer(0));
        self.state.const_blobs.push((binding_idx, Box::new(consts)));

        self
    }

    pub fn dynamic_storage_buffer_vec<T: Copy + 'static>(mut self, consts: Vec<T>) -> Self {
        let binding_idx = self.state.bindings.len();

        self.state
            .bindings
            .push(RenderPassBinding::DynamicConstantsStorageBuffer(0));
        self.state
            .const_blobs
            .push((binding_idx, Box::new(VecBlob(consts))));

        self
    }

    pub fn raw_descriptor_set(mut self, set_idx: u32, set: vk::DescriptorSet) -> Self {
        self.state.raw_descriptor_sets.push((set_idx, set));
        self
    }

    pub fn bind<Binding>(self, binding: &Binding) -> Self
    where
        Binding: BindToSimpleRenderPass<'rg, RgPipelineHandle>,
    {
        binding.bind(self)
    }

    pub fn bind_mut<Binding>(self, binding: &mut Binding) -> Self
    where
        Binding: BindMutToSimpleRenderPass<'rg, RgPipelineHandle>,
    {
        binding.bind_mut(self)
    }
}

pub trait BindToSimpleRenderPass<'rg, RgPipelineHandle> {
    fn bind(
        &self,
        pass: SimpleRenderPass<'rg, RgPipelineHandle>,
    ) -> SimpleRenderPass<'rg, RgPipelineHandle>;
}

pub trait BindMutToSimpleRenderPass<'rg, RgPipelineHandle> {
    fn bind_mut(
        &mut self,
        pass: SimpleRenderPass<'rg, RgPipelineHandle>,
    ) -> SimpleRenderPass<'rg, RgPipelineHandle>;
}
