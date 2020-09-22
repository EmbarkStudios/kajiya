use ash::vk;

use super::{
    graph::RenderGraphExecutionParams, resource::*, RgComputePipelineHandle, RgRasterPipelineHandle,
};
use crate::{
    backend::image::ImageViewDesc, backend::shader::ComputePipeline,
    backend::shader::RasterPipeline, dynamic_constants::DynamicConstants,
    pipeline_cache::ComputePipelineHandle, pipeline_cache::RasterPipelineHandle,
};
use std::sync::Arc;

pub enum AnyRenderResource {
    OwnedImage(crate::backend::image::Image),
    ImportedImage(Arc<crate::backend::image::Image>),
    OwnedBuffer(crate::backend::buffer::Buffer),
    ImportedBuffer(Arc<crate::backend::buffer::Buffer>),
}

impl AnyRenderResource {
    pub fn borrow(&self) -> AnyRenderResourceRef {
        match self {
            AnyRenderResource::OwnedImage(inner) => AnyRenderResourceRef::Image(inner),
            AnyRenderResource::ImportedImage(inner) => AnyRenderResourceRef::Image(&*inner),
            AnyRenderResource::OwnedBuffer(inner) => AnyRenderResourceRef::Buffer(inner),
            AnyRenderResource::ImportedBuffer(inner) => AnyRenderResourceRef::Buffer(&*inner),
        }
    }
}

pub enum AnyRenderResourceRef<'a> {
    Image(&'a crate::backend::image::Image),
    Buffer(&'a crate::backend::buffer::Buffer),
}

pub struct ResourceRegistry<'exec_params, 'constants> {
    pub execution_params: &'exec_params RenderGraphExecutionParams<'exec_params>,
    pub(crate) resources: Vec<AnyRenderResource>,
    pub dynamic_constants: &'constants mut DynamicConstants,
    pub compute_pipelines: Vec<ComputePipelineHandle>,
    pub raster_pipelines: Vec<RasterPipelineHandle>,
}

impl<'exec_params, 'constants> ResourceRegistry<'exec_params, 'constants> {
    pub(crate) fn image<ViewType: GpuViewType>(&self, resource: Ref<Image, ViewType>) -> &Image {
        self.image_from_raw_handle::<ViewType>(resource.handle)
    }

    pub(crate) fn image_from_raw_handle<ViewType: GpuViewType>(
        &self,
        handle: GraphRawResourceHandle,
    ) -> &Image {
        match &self.resources[handle.id as usize].borrow() {
            AnyRenderResourceRef::Image(img) => *img,
            _ => panic!(),
        }
    }

    pub(crate) fn buffer<ViewType: GpuViewType>(&self, resource: Ref<Buffer, ViewType>) -> &Buffer {
        self.buffer_from_raw_handle::<ViewType>(resource.handle)
    }

    pub(crate) fn buffer_from_raw_handle<ViewType: GpuViewType>(
        &self,
        handle: GraphRawResourceHandle,
    ) -> &Buffer {
        match &self.resources[handle.id as usize].borrow() {
            AnyRenderResourceRef::Buffer(buffer) => *buffer,
            _ => panic!(),
        }
    }

    pub(crate) fn image_view<'a, 's>(
        &'s self,
        resource: GraphRawResourceHandle,
        view_desc: &ImageViewDesc,
    ) -> vk::ImageView
    where
        's: 'a,
    {
        let view_desc = view_desc;

        let image = match &self.resources[resource.id as usize].borrow() {
            AnyRenderResourceRef::Image(img) => *img,
            AnyRenderResourceRef::Buffer(_) => panic!(),
        };

        let device = self.execution_params.device;
        image.view(device, view_desc)
    }

    pub fn compute_pipeline(&self, pipeline: RgComputePipelineHandle) -> Arc<ComputePipeline> {
        let handle = self.compute_pipelines[pipeline.id];
        self.execution_params.pipeline_cache.get_compute(handle)
    }

    pub fn raster_pipeline(&self, pipeline: RgRasterPipelineHandle) -> Arc<RasterPipeline> {
        let handle = self.raster_pipelines[pipeline.id];
        self.execution_params.pipeline_cache.get_raster(handle)
    }
}
