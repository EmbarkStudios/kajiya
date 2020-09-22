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
}

impl AnyRenderResource {
    pub fn borrow(&self) -> AnyRenderResourceRef {
        match self {
            AnyRenderResource::OwnedImage(inner) => AnyRenderResourceRef::Image(inner),
            AnyRenderResource::ImportedImage(inner) => AnyRenderResourceRef::Image(&*inner),
            AnyRenderResource::OwnedBuffer(inner) => AnyRenderResourceRef::Buffer(inner),
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
    /*pub fn resource<'a, 's, ResType: Resource, ViewType>(
        &'s self,
        resource: Ref<ResType, ViewType>,
    ) -> GpuResourceView<'a, ResType, ViewType>
    where
        ViewType: GpuViewType,
        's: 'a,
    {
        // println!("ResourceRegistry::get: {:?}", resource.handle);
        GpuResourceView::<'a, ResType, ViewType>::new(<ResType as Resource>::borrow_resource(
            &self.resources[resource.handle.id as usize],
        ))
    }*/

    pub(crate) fn image<ViewType: GpuViewType>(&self, resource: Ref<Image, ViewType>) -> &Image {
        match &self.resources[resource.handle.id as usize].borrow() {
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

    /*pub fn render_pass(
        &self,
        render_target: &RenderTarget,
    ) -> anyhow::Result<RenderResourceHandle> {
        let device = self.execution_params.device;

        let frame_binding_set_handle = self
            .execution_params
            .handles
            .allocate_transient(RenderResourceType::FrameBindingSet);

        let mut render_target_views = [None; MAX_RENDER_TARGET_COUNT];
        for (i, rt) in render_target.color.iter().enumerate() {
            if let Some(rt) = rt {
                render_target_views[i] = Some(RenderBindingRenderTargetView {
                    base: RenderBindingView {
                        resource: self.resources[rt.texture.handle.id as usize],
                        format: rt.texture.desc().format,
                        dimension: RenderViewDimension::Tex2d,
                    },
                    mip_slice: 0,
                    first_array_slice: 0,
                    plane_slice_first_w_slice: 0,
                    array_size: 0,
                    w_size: 0,
                });
            }
        }

        device.create_frame_binding_set(
            frame_binding_set_handle,
            &RenderFrameBindingSetDesc {
                render_target_views,
                depth_stencil_view: None,
            },
            "draw binding set".into(),
        )?;

        let render_pass_handle = self
            .execution_params
            .handles
            .allocate_transient(RenderResourceType::RenderPass);

        device.create_render_pass(
            render_pass_handle,
            &RenderPassDesc {
                frame_binding: frame_binding_set_handle,
                // TODO
                render_target_info: [RenderTargetInfo {
                    load_op: RenderLoadOp::Discard,
                    store_op: RenderStoreOp::Store,
                    clear_color: [0.0f32; 4],
                }; MAX_RENDER_TARGET_COUNT],
                depth_stencil_target_info: DepthStencilTargetInfo {
                    load_op: RenderLoadOp::Discard,
                    store_op: RenderStoreOp::Discard,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                },
            },
            "render pass".into(),
        )?;

        Ok(render_pass_handle)
    }

    pub fn raster_pipeline(
        &self,
        desc: RasterPipelineDesc,
        render_target: &RenderTarget,
    ) -> anyhow::Result<Arc<RasterPipeline>> {
        self.execution_params.pipeline_cache.get_or_load_raster(
            self.execution_params,
            desc,
            render_target,
        )
    }*/
}
