use super::{
    graph::{GraphResourceCreateInfo, RecordedPass, RenderGraph},
    resource::*,
    PassResourceAccessType, PassResourceRef, RenderPassApi, RgComputePipeline,
    RgComputePipelineHandle, RgRasterPipeline, RgRasterPipelineHandle, RgRtPipeline,
    RgRtPipelineHandle,
};

use crate::backend::{
    ray_tracing::RayTracingPipelineDesc,
    shader::{
        ComputePipelineDesc, DescriptorSetLayoutOpts, PipelineShader, RasterPipelineDescBuilder,
    },
};
use std::{marker::PhantomData, path::Path};

pub use vk_sync::AccessType;

pub struct PassBuilder<'rg> {
    pub(crate) rg: &'rg mut RenderGraph,
    pub(crate) pass_idx: usize,
    pub(crate) pass: Option<RecordedPass>,
}

impl<'s> Drop for PassBuilder<'s> {
    fn drop(&mut self) {
        self.rg.record_pass(self.pass.take().unwrap())
    }
}

pub trait TypeEquals {
    type Other;
    fn same(value: Self) -> Self::Other;
}

impl<T: Sized> TypeEquals for T {
    type Other = Self;
    fn same(value: Self) -> Self::Other {
        value
    }
}

#[allow(dead_code)]
impl<'rg> PassBuilder<'rg> {
    pub fn create<Desc: ResourceDesc>(
        &mut self,
        desc: &Desc,
    ) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        let handle: Handle<<Desc as ResourceDesc>::Resource> = Handle {
            raw: self.rg.create_raw_resource(GraphResourceCreateInfo {
                desc: desc.clone().into(),
                create_pass_idx: self.pass_idx,
            }),
            desc: TypeEquals::same(desc.clone()),
            marker: PhantomData,
        };

        handle
    }

    pub fn write_impl<Res: Resource, ViewType: GpuViewType>(
        &mut self,
        handle: &mut Handle<Res>,
        access_type: vk_sync::AccessType,
    ) -> Ref<Res, ViewType> {
        let pass = self.pass.as_mut().unwrap();

        // Don't know of a good way to use the borrow checker to verify that writes and reads
        // don't overlap, and that multiple writes don't happen to the same resource.
        // The borrow checker will at least check that resources don't alias each other,
        // but for the access in render passes, we resort to a runtime check.
        if pass.write.iter().any(|item| item.handle == handle.raw) {
            panic!("Trying to write twice to the same resource within one render pass");
        } else if pass.read.iter().any(|item| item.handle == handle.raw) {
            panic!("Trying to read and write to the same resource within one render pass");
        }

        pass.write.push(PassResourceRef {
            handle: handle.raw,
            access: PassResourceAccessType::new(access_type),
        });

        Ref {
            desc: handle.desc.clone(),
            handle: handle.raw.next_version(),
            marker: PhantomData,
        }
    }

    pub fn write<Res: Resource>(
        &mut self,
        handle: &mut Handle<Res>,
        access_type: vk_sync::AccessType,
    ) -> Ref<Res, GpuUav> {
        match access_type {
            AccessType::CommandBufferWriteNVX
            | AccessType::VertexShaderWrite
            | AccessType::TessellationControlShaderWrite
            | AccessType::TessellationEvaluationShaderWrite
            | AccessType::GeometryShaderWrite
            | AccessType::FragmentShaderWrite
            | AccessType::ComputeShaderWrite
            | AccessType::AnyShaderWrite
            | AccessType::TransferWrite
            | AccessType::HostWrite
            | AccessType::ColorAttachmentReadWrite
            | AccessType::General => {}
            _ => {
                panic!("Invalid access type: {:?}", access_type);
            }
        }

        self.write_impl(handle, access_type)
    }

    pub fn raster<Res: Resource>(
        &mut self,
        handle: &mut Handle<Res>,
        access_type: vk_sync::AccessType,
    ) -> Ref<Res, GpuRt> {
        match access_type {
            AccessType::ColorAttachmentWrite
            | AccessType::DepthStencilAttachmentWrite
            | AccessType::DepthAttachmentWriteStencilReadOnly
            | AccessType::StencilAttachmentWriteDepthReadOnly => {}
            _ => {
                panic!("Invalid access type: {:?}", access_type);
            }
        }

        self.write_impl(handle, access_type)
    }

    pub fn read<Res: Resource>(
        &mut self,
        handle: &Handle<Res>,
        access_type: vk_sync::AccessType,
    ) -> Ref<Res, GpuSrv> {
        match access_type {
            AccessType::CommandBufferReadNVX
            | AccessType::IndirectBuffer
            | AccessType::IndexBuffer
            | AccessType::VertexBuffer
            | AccessType::VertexShaderReadUniformBuffer
            | AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::VertexShaderReadOther
            | AccessType::TessellationControlShaderReadUniformBuffer
            | AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::TessellationControlShaderReadOther
            | AccessType::TessellationEvaluationShaderReadUniformBuffer
            | AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::TessellationEvaluationShaderReadOther
            | AccessType::GeometryShaderReadUniformBuffer
            | AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::GeometryShaderReadOther
            | AccessType::FragmentShaderReadUniformBuffer
            | AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::FragmentShaderReadColorInputAttachment
            | AccessType::FragmentShaderReadDepthStencilInputAttachment
            | AccessType::FragmentShaderReadOther
            | AccessType::ColorAttachmentRead
            | AccessType::DepthStencilAttachmentRead
            | AccessType::ComputeShaderReadUniformBuffer
            | AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::ComputeShaderReadOther
            | AccessType::AnyShaderReadUniformBuffer
            | AccessType::AnyShaderReadUniformBufferOrVertexBuffer
            | AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::AnyShaderReadOther
            | AccessType::TransferRead
            | AccessType::HostRead
            | AccessType::Present => {}
            _ => {
                panic!("Invalid access type: {:?}", access_type);
            }
        }

        let pass = self.pass.as_mut().unwrap();

        // Runtime "borrow" check; see info in `write` above.
        if pass.write.iter().any(|item| item.handle == handle.raw) {
            panic!("Trying to read and write to the same resource within one render pass");
        }

        pass.write.push(PassResourceRef {
            handle: handle.raw,
            access: PassResourceAccessType::new(access_type),
        });

        Ref {
            desc: handle.desc.clone(),
            handle: handle.raw,
            marker: PhantomData,
        }
    }

    pub fn register_compute_pipeline(&mut self, path: impl AsRef<Path>) -> RgComputePipelineHandle {
        let id = self.rg.compute_pipelines.len();

        let mut desc = ComputePipelineDesc::builder().build().unwrap();

        for (set_idx, layout) in &self.rg.predefined_descriptor_set_layouts {
            desc.descriptor_set_opts[*set_idx as usize] = Some((
                *set_idx,
                DescriptorSetLayoutOpts::builder()
                    .replace(layout.bindings.clone())
                    .build()
                    .unwrap(),
            ));
        }

        self.rg.compute_pipelines.push(RgComputePipeline {
            shader_path: path.as_ref().to_owned(),
            desc,
        });

        RgComputePipelineHandle { id }
    }

    pub fn register_raster_pipeline(
        &mut self,
        shaders: &[PipelineShader<&'static str>],
        desc: RasterPipelineDescBuilder,
    ) -> RgRasterPipelineHandle {
        let id = self.rg.raster_pipelines.len();
        let mut desc = desc.build().unwrap();

        for (set_idx, layout) in &self.rg.predefined_descriptor_set_layouts {
            desc.descriptor_set_opts[*set_idx as usize] = Some((
                *set_idx,
                DescriptorSetLayoutOpts::builder()
                    .replace(layout.bindings.clone())
                    .build()
                    .unwrap(),
            ));
        }

        self.rg.raster_pipelines.push(RgRasterPipeline {
            shaders: shaders
                .iter()
                .map(|shader| {
                    let desc = shader.desc.clone();

                    PipelineShader {
                        code: shader.code,
                        desc,
                    }
                })
                .collect(),
            desc,
        });

        RgRasterPipelineHandle { id }
    }

    pub fn register_ray_tracing_pipeline(
        &mut self,
        shaders: &[PipelineShader<&'static str>],
        mut desc: RayTracingPipelineDesc,
    ) -> RgRtPipelineHandle {
        let id = self.rg.rt_pipelines.len();

        for (set_idx, layout) in &self.rg.predefined_descriptor_set_layouts {
            desc.descriptor_set_opts[*set_idx as usize] = Some((
                *set_idx,
                DescriptorSetLayoutOpts::builder()
                    .replace(layout.bindings.clone())
                    .build()
                    .unwrap(),
            ));
        }

        self.rg.rt_pipelines.push(RgRtPipeline {
            shaders: shaders
                .iter()
                .map(|shader| {
                    let desc = shader.desc.clone();

                    PipelineShader {
                        code: shader.code,
                        desc,
                    }
                })
                .collect(),
            desc,
        });

        RgRtPipelineHandle { id }
    }

    pub fn render(mut self, render: impl FnOnce(&mut RenderPassApi) + 'static) {
        let prev = self
            .pass
            .as_mut()
            .unwrap()
            .render_fn
            .replace(Box::new(render));

        assert!(prev.is_none());
    }
}
