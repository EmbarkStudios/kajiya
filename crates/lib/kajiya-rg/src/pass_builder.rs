use crate::{PassResourceAccessSyncType, RenderPassApi};

use super::{
    graph::{
        PassResourceAccessType, PassResourceRef, RecordedPass, RenderGraph, RgComputePipeline,
        RgComputePipelineHandle, RgRasterPipeline, RgRasterPipelineHandle, RgRtPipeline,
        RgRtPipelineHandle, TypeEquals,
    },
    resource::*,
};

use kajiya_backend::{
    vk_sync::{self, AccessType},
    vulkan::{ray_tracing::RayTracingPipelineDesc, shader::*},
    BackendError,
};
use std::{marker::PhantomData, path::Path};

pub struct PassBuilder<'rg> {
    pub(crate) rg: &'rg mut RenderGraph,
    #[allow(dead_code)]
    pub(crate) pass_idx: usize,
    pub(crate) pass: Option<RecordedPass>,
}

impl<'s> Drop for PassBuilder<'s> {
    fn drop(&mut self) {
        self.rg.record_pass(self.pass.take().unwrap())
    }
}

impl<'rg> PassBuilder<'rg> {
    pub fn create<Desc: ResourceDesc>(
        &mut self,
        desc: Desc,
    ) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        self.rg.create(desc)
    }

    pub fn write_impl<Res: Resource, ViewType: GpuViewType>(
        &mut self,
        handle: &mut Handle<Res>,
        access_type: vk_sync::AccessType,
        sync_type: PassResourceAccessSyncType,
    ) -> Ref<Res, ViewType> {
        let pass = self.pass.as_mut().unwrap();

        // CHECK DISABLED: multiple writes or mixing of reads and writes is valid with non-overlapping views
        /*
        // Don't know of a good way to use the borrow checker to verify that writes and reads
        // don't overlap, and that multiple writes don't happen to the same resource.
        // The borrow checker will at least check that resources don't alias each other,
        // but for the access in render passes, we resort to a runtime check.
        if pass.write.iter().any(|item| item.handle == handle.raw) {
            panic!("Trying to write twice to the same resource within one render pass");
        } else if pass.read.iter().any(|item| item.handle == handle.raw) {
            panic!("Trying to read and write to the same resource within one render pass");
        }*/

        pass.write.push(PassResourceRef {
            handle: handle.raw,
            access: PassResourceAccessType::new(access_type, sync_type),
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

        self.write_impl(handle, access_type, PassResourceAccessSyncType::AlwaysSync)
    }

    pub fn write_no_sync<Res: Resource>(
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

        self.write_impl(
            handle,
            access_type,
            PassResourceAccessSyncType::SkipSyncIfSameAccessType,
        )
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

        self.write_impl(handle, access_type, PassResourceAccessSyncType::AlwaysSync)
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

        // CHECK DISABLED: multiple writes or mixing of reads and writes is valid with non-overlapping views
        /*// Runtime "borrow" check; see info in `write` above.
        if pass.write.iter().any(|item| item.handle == handle.raw) {
            panic!("Trying to read and write to the same resource within one render pass");
        }*/

        pass.read.push(PassResourceRef {
            handle: handle.raw,
            access: PassResourceAccessType::new(
                access_type,
                PassResourceAccessSyncType::SkipSyncIfSameAccessType,
            ),
        });

        Ref {
            desc: handle.desc.clone(),
            handle: handle.raw,
            marker: PhantomData,
        }
    }

    pub fn raster_read<Res: Resource>(
        &mut self,
        handle: &Handle<Res>,
        access_type: vk_sync::AccessType,
    ) -> Ref<Res, GpuRt> {
        match access_type {
            AccessType::ColorAttachmentRead | AccessType::DepthStencilAttachmentRead => {}
            _ => {
                panic!("Invalid access type: {:?}", access_type);
            }
        }

        let pass = self.pass.as_mut().unwrap();

        pass.read.push(PassResourceRef {
            handle: handle.raw,
            access: PassResourceAccessType::new(
                access_type,
                PassResourceAccessSyncType::SkipSyncIfSameAccessType,
            ),
        });

        Ref {
            desc: handle.desc.clone(),
            handle: handle.raw,
            marker: PhantomData,
        }
    }

    pub fn register_compute_pipeline(&mut self, path: impl AsRef<Path>) -> RgComputePipelineHandle {
        let desc = ComputePipelineDesc::builder()
            .compute_hlsl(path.as_ref().to_owned())
            .build()
            .unwrap();
        self.register_compute_pipeline_with_desc(desc)
    }

    pub fn register_compute_pipeline_with_desc(
        &mut self,
        mut desc: ComputePipelineDesc,
    ) -> RgComputePipelineHandle {
        let id = self.rg.compute_pipelines.len();

        for (set_idx, layout) in &self.rg.predefined_descriptor_set_layouts {
            desc.descriptor_set_opts[*set_idx as usize] = Some((
                *set_idx,
                DescriptorSetLayoutOpts::builder()
                    .replace(layout.bindings.clone())
                    .build()
                    .unwrap(),
            ));
        }

        self.rg.compute_pipelines.push(RgComputePipeline { desc });

        RgComputePipelineHandle { id }
    }

    pub fn register_raster_pipeline(
        &mut self,
        shaders: &[PipelineShaderDesc],
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
            shaders: shaders.to_vec(),
            desc,
        });

        RgRasterPipelineHandle { id }
    }

    pub fn register_ray_tracing_pipeline(
        &mut self,
        shaders: &[PipelineShaderDesc],
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
            shaders: shaders.to_vec(),
            desc,
        });

        RgRtPipelineHandle { id }
    }

    pub fn render(
        mut self,
        render: impl (FnOnce(&mut RenderPassApi) -> Result<(), BackendError>) + 'static,
    ) {
        let prev = self
            .pass
            .as_mut()
            .unwrap()
            .render_fn
            .replace(Box::new(render));

        assert!(prev.is_none());
    }
}
