use crate::{GraphResourceInfo, RenderGraphPipelines};

use super::{
    graph::RenderGraphExecutionParams, resource::*, RgComputePipelineHandle,
    RgRasterPipelineHandle, RgRtPipelineHandle,
};
use kajiya_backend::{
    ash::vk,
    dynamic_constants::DynamicConstants,
    vk_sync,
    vulkan::{
        ray_tracing::{RayTracingAcceleration, RayTracingPipeline},
        shader::{ComputePipeline, RasterPipeline},
    },
    BackendError,
};
use std::sync::Arc;

pub struct PendingRenderResourceInfo {
    pub(crate) resource: GraphResourceInfo,
}

pub enum AnyRenderResource {
    OwnedImage(Image),
    ImportedImage(Arc<Image>),
    OwnedBuffer(Buffer),
    ImportedBuffer(Arc<Buffer>),
    ImportedRayTracingAcceleration(Arc<RayTracingAcceleration>),

    // Must be replaced before access. Used to late-update swapchain resources.
    Pending(PendingRenderResourceInfo),
}

impl AnyRenderResource {
    #[track_caller]
    pub fn borrow(&self) -> AnyRenderResourceRef {
        match self {
            AnyRenderResource::OwnedImage(inner) => AnyRenderResourceRef::Image(inner),
            AnyRenderResource::ImportedImage(inner) => AnyRenderResourceRef::Image(inner.as_ref()),
            AnyRenderResource::OwnedBuffer(inner) => AnyRenderResourceRef::Buffer(inner),
            AnyRenderResource::ImportedBuffer(inner) => {
                AnyRenderResourceRef::Buffer(inner.as_ref())
            }
            AnyRenderResource::ImportedRayTracingAcceleration(inner) => {
                AnyRenderResourceRef::RayTracingAcceleration(inner.as_ref())
            }
            AnyRenderResource::Pending { .. } => {
                panic!("AnyRenderResource::borrow called while the resource was in Pending state")
            }
        }
    }
}

pub enum AnyRenderResourceRef<'a> {
    Image(&'a Image),
    Buffer(&'a Buffer),
    RayTracingAcceleration(&'a RayTracingAcceleration),
}

pub(crate) struct RegistryResource {
    pub resource: AnyRenderResource,
    pub access_type: vk_sync::AccessType,
}

pub struct ResourceRegistry<'exec_params, 'constants> {
    pub execution_params: RenderGraphExecutionParams<'exec_params>,
    pub(crate) resources: Vec<RegistryResource>,
    pub dynamic_constants: &'constants mut DynamicConstants,
    pub pipelines: RenderGraphPipelines,
}

impl<'exec_params, 'constants> ResourceRegistry<'exec_params, 'constants> {
    pub fn image<ViewType: GpuViewType>(&self, resource: Ref<Image, ViewType>) -> &Image {
        self.image_from_raw_handle::<ViewType>(resource.handle)
    }

    pub(crate) fn image_from_raw_handle<ViewType: GpuViewType>(
        &self,
        handle: GraphRawResourceHandle,
    ) -> &Image {
        match &self.resources[handle.id as usize].resource.borrow() {
            AnyRenderResourceRef::Image(img) => img,
            _ => panic!(),
        }
    }

    pub fn buffer<ViewType: GpuViewType>(&self, resource: Ref<Buffer, ViewType>) -> &Buffer {
        self.buffer_from_raw_handle::<ViewType>(resource.handle)
    }

    pub(crate) fn buffer_from_raw_handle<ViewType: GpuViewType>(
        &self,
        handle: GraphRawResourceHandle,
    ) -> &Buffer {
        match &self.resources[handle.id as usize].resource.borrow() {
            AnyRenderResourceRef::Buffer(buffer) => buffer,
            _ => panic!(),
        }
    }

    pub fn rt_acceleration<ViewType: GpuViewType>(
        &self,
        resource: Ref<RayTracingAcceleration, ViewType>,
    ) -> &RayTracingAcceleration {
        self.rt_acceleration_from_raw_handle::<ViewType>(resource.handle)
    }

    pub(crate) fn rt_acceleration_from_raw_handle<ViewType: GpuViewType>(
        &self,
        handle: GraphRawResourceHandle,
    ) -> &RayTracingAcceleration {
        match &self.resources[handle.id as usize].resource.borrow() {
            AnyRenderResourceRef::RayTracingAcceleration(acc) => acc,
            _ => panic!(),
        }
    }

    pub(crate) fn image_view<'a, 's>(
        &'s self,
        resource: GraphRawResourceHandle,
        view_desc: &ImageViewDesc,
    ) -> Result<vk::ImageView, BackendError>
    where
        's: 'a,
    {
        let view_desc = view_desc;

        let image = match &self.resources[resource.id as usize].resource.borrow() {
            AnyRenderResourceRef::Image(img) => *img,
            _ => panic!(),
        };

        let device = self.execution_params.device;
        image.view(device, view_desc)
    }

    pub fn compute_pipeline(&self, pipeline: RgComputePipelineHandle) -> Arc<ComputePipeline> {
        let handle = self.pipelines.compute[pipeline.id];
        self.execution_params.pipeline_cache.get_compute(handle)
    }

    pub fn raster_pipeline(&self, pipeline: RgRasterPipelineHandle) -> Arc<RasterPipeline> {
        let handle = self.pipelines.raster[pipeline.id];
        self.execution_params.pipeline_cache.get_raster(handle)
    }

    pub fn ray_tracing_pipeline(&self, pipeline: RgRtPipelineHandle) -> Arc<RayTracingPipeline> {
        let handle = self.pipelines.rt[pipeline.id];
        self.execution_params.pipeline_cache.get_ray_tracing(handle)
    }
}
