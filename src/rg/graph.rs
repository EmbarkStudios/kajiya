#![allow(unused_imports)]

use super::{
    pass_builder::PassBuilder,
    resource::*,
    resource_registry::{AnyRenderResource, ResourceRegistry},
    RenderPassApi,
};

use crate::{
    backend::barrier::get_access_info,
    backend::barrier::record_image_barrier,
    backend::barrier::ImageBarrier,
    backend::device::{CommandBuffer, Device},
    backend::image::ImageView,
    backend::image::ImageViewDesc,
    backend::shader::ComputePipelineDesc,
    dynamic_constants::DynamicConstants,
    pipeline_cache::ComputePipelineHandle,
    pipeline_cache::PipelineCache,
    view_cache::ViewCache,
};
use ash::vk;
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    hash::Hash,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
    sync::Weak,
};

pub(crate) struct GraphResourceCreateInfo {
    pub desc: GraphResourceDesc,
    pub create_pass_idx: usize,
}

#[derive(Clone, Copy)]
pub struct RgComputePipelineHandle {
    pub(crate) id: usize,
}

pub(crate) struct RgComputePipeline {
    pub(crate) shader_path: PathBuf,
    pub(crate) desc: ComputePipelineDesc,
}

pub struct RenderGraph {
    passes: Vec<RecordedPass>,
    resources: Vec<GraphResourceCreateInfo>,
    pub(crate) compute_pipelines: Vec<RgComputePipeline>,
    pub(crate) frame_descriptor_set_layout: Option<HashMap<u32, rspirv_reflect::DescriptorInfo>>,
}

impl RenderGraph {
    pub fn new(
        frame_descriptor_set_layout: Option<HashMap<u32, rspirv_reflect::DescriptorInfo>>,
    ) -> Self {
        Self {
            passes: Vec::new(),
            resources: Vec::new(),
            compute_pipelines: Vec::new(),
            frame_descriptor_set_layout,
        }
    }

    pub(crate) fn create_raw_resource(
        &mut self,
        info: GraphResourceCreateInfo,
    ) -> GraphRawResourceHandle {
        let res = GraphRawResourceHandle {
            id: self.resources.len() as u32,
            version: 0,
        };

        self.resources.push(info);
        res
    }
}

#[derive(Debug)]
struct ResourceLifetime {
    first_access: usize,
    last_access: usize,
}

struct ResourceInfo {
    lifetimes: Vec<ResourceLifetime>,
    image_usage_flags: Vec<vk::ImageUsageFlags>,
}

pub struct RenderGraphExecutionParams<'a> {
    pub device: &'a Device,
    pub pipeline_cache: &'a mut PipelineCache,
    pub view_cache: &'a ViewCache,
    pub frame_descriptor_set: vk::DescriptorSet,
    pub frame_constants_offset: u32,
}

pub struct CompiledRenderGraph {
    rg: RenderGraph,
    resources: Vec<AnyRenderResource>,
    compute_pipelines: Vec<ComputePipelineHandle>,
}

impl RenderGraph {
    pub fn add_pass<'s>(&'s mut self) -> PassBuilder<'s> {
        let pass_idx = self.passes.len();

        PassBuilder {
            rg: self,
            pass_idx,
            pass: Some(Default::default()),
        }
    }

    fn calculate_resource_info(&self) -> ResourceInfo {
        let mut lifetimes: Vec<ResourceLifetime> = self
            .resources
            .iter()
            .map(|res| ResourceLifetime {
                first_access: res.create_pass_idx,
                last_access: res.create_pass_idx,
            })
            .collect();

        let mut image_usage_flags: Vec<vk::ImageUsageFlags> =
            vec![Default::default(); self.resources.len()];

        for (pass_idx, pass) in self.passes.iter().enumerate() {
            for res_access in pass.read.iter().chain(pass.write.iter()) {
                let res = &mut lifetimes[res_access.handle.id as usize];
                res.last_access = res.last_access.max(pass_idx);

                let access_mask = get_access_info(res_access.access.access_type).access_mask;
                let image_usage: vk::ImageUsageFlags = match access_mask {
                    vk::AccessFlags::SHADER_READ => vk::ImageUsageFlags::SAMPLED,
                    vk::AccessFlags::SHADER_WRITE => vk::ImageUsageFlags::STORAGE,
                    vk::AccessFlags::COLOR_ATTACHMENT_READ => vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE => {
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                    }
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ => {
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    }
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE => {
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    }
                    vk::AccessFlags::TRANSFER_READ => vk::ImageUsageFlags::TRANSFER_SRC,
                    vk::AccessFlags::TRANSFER_WRITE => vk::ImageUsageFlags::TRANSFER_DST,
                    _ => panic!("Invalid image access mask: {:?}", access_mask),
                };

                image_usage_flags[res_access.handle.id as usize] |= image_usage;
            }
        }

        ResourceInfo {
            lifetimes,
            image_usage_flags,
        }
    }

    pub fn compile(
        self,
        device: &Device,
        pipeline_cache: &mut PipelineCache,
    ) -> CompiledRenderGraph {
        let resource_info = self.calculate_resource_info();
        // TODO: alias resources

        /* println!(
            "Resources: {:#?}",
            self.resources
                .iter()
                .map(|info| info.desc)
                .zip(resource_lifetimes.iter())
                .collect::<Vec<_>>()
        ); */

        let gpu_resources: Vec<AnyRenderResource> = self
            .resources
            .iter()
            .zip(resource_info.image_usage_flags.iter())
            .map(
                |(resource, image_usage_flags): (
                    &GraphResourceCreateInfo,
                    &vk::ImageUsageFlags,
                )| {
                    match resource.desc {
                        GraphResourceDesc::Image(mut desc) => {
                            desc.usage = *image_usage_flags;
                            AnyRenderResource::Image(device.create_image(desc, None).unwrap())
                        }
                    }
                },
            )
            .collect();

        let compute_pipelines = self
            .compute_pipelines
            .iter()
            .map(|pipeline| pipeline_cache.register_compute(&pipeline.shader_path, &pipeline.desc))
            .collect::<Vec<_>>();

        CompiledRenderGraph {
            rg: self,
            resources: gpu_resources,
            compute_pipelines,
        }
    }

    pub(crate) fn record_pass(&mut self, pass: RecordedPass) {
        self.passes.push(pass);
    }
}

impl CompiledRenderGraph {
    pub fn execute(
        self,
        params: RenderGraphExecutionParams<'_>,
        dynamic_constants: &mut DynamicConstants,
        cb: &CommandBuffer,
    ) -> anyhow::Result<()> {
        let mut resource_registry = ResourceRegistry {
            execution_params: &params,
            resources: self.resources,
            dynamic_constants: dynamic_constants,
            compute_pipelines: self.compute_pipelines,
        };

        for pass in self.rg.passes.into_iter() {
            {
                let mut transitions: Vec<(&AnyRenderResource, PassResourceAccessType)> = Vec::new();
                for resource_ref in pass.read.iter().chain(pass.write.iter()) {
                    transitions.push((
                        &resource_registry.resources[resource_ref.handle.id as usize],
                        resource_ref.access,
                    ));
                }

                // TODO: optimize the barriers

                for (resource, access) in transitions {
                    match resource {
                        AnyRenderResource::Image(image) => {
                            record_image_barrier(
                                &params.device.raw,
                                cb.raw,
                                ImageBarrier::new(
                                    image.raw,
                                    vk_sync::AccessType::Nothing, // TODO
                                    access.access_type,
                                    vk::ImageAspectFlags::COLOR, // TODO
                                ),
                            );
                        }
                        AnyRenderResource::Buffer(_) => {
                            todo!();
                        }
                    }
                }
            }

            let mut api = RenderPassApi {
                cb,
                resources: &mut resource_registry,
            };

            (pass.render_fn.unwrap())(&mut api);
        }

        Ok(())
    }
}

type DynRenderFn = dyn FnOnce(&mut RenderPassApi);

#[derive(Copy, Clone)]
pub struct PassResourceAccessType {
    // TODO: multiple
    access_type: vk_sync::AccessType,
}

impl PassResourceAccessType {
    pub fn new(access_type: vk_sync::AccessType) -> Self {
        Self { access_type }
    }
}

pub(crate) struct PassResourceRef {
    pub handle: GraphRawResourceHandle,
    pub access: PassResourceAccessType,
}

#[derive(Default)]
pub(crate) struct RecordedPass {
    pub read: Vec<PassResourceRef>,
    pub write: Vec<PassResourceRef>,
    pub render_fn: Option<Box<DynRenderFn>>,
}
