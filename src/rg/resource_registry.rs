use ash::version::DeviceV1_0;

use super::{
    graph::RenderGraphExecutionParams, resource::*, ImageViewCacheKey, RgComputePipelineHandle,
};
use crate::{
    backend::device::CommandBuffer, backend::image::ImageView, backend::image::ImageViewDesc,
    backend::image::ImageViewDescBuilder, backend::shader::ShaderPipeline,
    dynamic_constants::DynamicConstants, pipeline_cache::ComputePipelineHandle,
};
use std::{path::Path, sync::Arc};

pub enum AnyRenderResource {
    Image(Arc<crate::backend::image::Image>),
    Buffer(crate::backend::buffer::Buffer),
}

pub struct ResourceRegistry<'exec_params, 'constants> {
    pub execution_params: &'exec_params RenderGraphExecutionParams<'exec_params>,
    pub(crate) resources: Vec<AnyRenderResource>,
    pub dynamic_constants: &'constants mut DynamicConstants,
    pub compute_pipelines: Vec<ComputePipelineHandle>,
}

impl<'exec_params, 'constants> ResourceRegistry<'exec_params, 'constants> {
    pub fn resource<'a, 's, ResType: Resource, ViewType>(
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
    }

    pub(crate) fn image_view<'a, 's>(
        &'s self,
        resource: GraphRawResourceHandle,
        view_desc: &ImageViewDesc,
    ) -> Arc<ImageView>
    where
        's: 'a,
    {
        let view_desc = view_desc;
        let mut views = self.execution_params.view_cache.image_views.lock();
        let views = &mut *views;

        let image = match &self.resources[resource.id as usize] {
            AnyRenderResource::Image(img) => img.clone(),
            AnyRenderResource::Buffer(_) => panic!(),
        };

        let key = ImageViewCacheKey {
            image: Arc::downgrade(&image),
            view_desc: view_desc.clone(),
        };
        let device = self.execution_params.device;

        views
            .entry(key)
            .or_insert_with(|| {
                Arc::new(device.create_image_view(view_desc.clone(), &image).unwrap())
            })
            .clone()
    }

    pub fn compute_pipeline(&self, pipeline: RgComputePipelineHandle) -> Arc<ShaderPipeline> {
        let handle = self.compute_pipelines[pipeline.0];
        self.execution_params.pipeline_cache.get_compute(handle)
    }

    pub fn bind_frame_constants(&self, cb: &CommandBuffer, shader: &ShaderPipeline) {
        if shader
            .set_layout_info
            .get(2)
            .map(|set| !set.is_empty())
            .unwrap_or_default()
        {
            unsafe {
                self.execution_params.device.raw.cmd_bind_descriptor_sets(
                    cb.raw,
                    shader.pipeline_bind_point,
                    shader.pipeline_layout,
                    2,
                    &[self.execution_params.frame_descriptor_set],
                    &[self.execution_params.frame_constants_offset],
                );
            }
        }
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
