#![allow(unused_imports)]

use super::{
    pass_builder::PassBuilder,
    resource::*,
    resource_registry::AnyRenderResourceRef,
    resource_registry::{AnyRenderResource, ResourceRegistry},
    RenderPassApi,
};

use crate::{
    backend::barrier::get_access_info,
    backend::barrier::image_aspect_mask_from_access_type_and_format,
    backend::barrier::record_image_barrier,
    backend::barrier::ImageBarrier,
    backend::device::{CommandBuffer, Device},
    backend::image::ImageViewDesc,
    backend::shader::ComputePipelineDesc,
    dynamic_constants::DynamicConstants,
    pipeline_cache::ComputePipelineHandle,
    pipeline_cache::PipelineCache,
    transient_resource_cache::TransientResourceCache,
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

pub(crate) enum GraphResourceImportInfo {
    Image(Arc<Image>),
}

pub(crate) enum GraphResourceInfo {
    Created(GraphResourceCreateInfo),
    Imported(GraphResourceImportInfo),
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
    resources: Vec<GraphResourceInfo>,
    exported_images: Vec<(GraphRawResourceHandle, vk::ImageUsageFlags)>,
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
            exported_images: Vec::new(),
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

        self.resources.push(GraphResourceInfo::Created(info));
        res
    }

    pub fn import_image(&mut self, img: Arc<Image>) -> Handle<Image> {
        let res = GraphRawResourceHandle {
            id: self.resources.len() as u32,
            version: 0,
        };

        let desc = img.desc;

        self.resources
            .push(GraphResourceInfo::Imported(GraphResourceImportInfo::Image(
                img,
            )));

        Handle {
            raw: res,
            desc,
            marker: PhantomData,
        }
    }

    pub fn export_image(
        &mut self,
        img: Handle<Image>,
        usage_flags: vk::ImageUsageFlags,
    ) -> ExportedHandle<Image> {
        self.exported_images.push((img.raw, usage_flags));
        ExportedHandle(img)
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
    pub frame_descriptor_set: vk::DescriptorSet,
    pub frame_constants_offset: u32,
}

pub struct CompiledRenderGraph {
    rg: RenderGraph,
    resource_info: ResourceInfo,
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
            .map(|res| match res {
                GraphResourceInfo::Created(res) => ResourceLifetime {
                    first_access: res.create_pass_idx,
                    last_access: res.create_pass_idx,
                },
                GraphResourceInfo::Imported(_) => ResourceLifetime {
                    first_access: 0,
                    last_access: 0,
                },
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

        for (res, usage) in self.exported_images.iter().copied() {
            let res = res.id as usize;
            lifetimes[res].last_access = self.passes.len().saturating_sub(1);
            image_usage_flags[res] |= usage;
        }

        ResourceInfo {
            lifetimes,
            image_usage_flags,
        }
    }

    pub fn compile(self, pipeline_cache: &mut PipelineCache) -> CompiledRenderGraph {
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

        let compute_pipelines = self
            .compute_pipelines
            .iter()
            .map(|pipeline| pipeline_cache.register_compute(&pipeline.shader_path, &pipeline.desc))
            .collect::<Vec<_>>();

        CompiledRenderGraph {
            rg: self,
            resource_info,
            compute_pipelines,
        }
    }

    pub(crate) fn record_pass(&mut self, pass: RecordedPass) {
        self.passes.push(pass);
    }
}

impl CompiledRenderGraph {
    #[must_use = "Call release_resources on the result"]
    pub fn execute(
        self,
        params: RenderGraphExecutionParams<'_>,
        transient_resource_cache: &mut TransientResourceCache,
        dynamic_constants: &mut DynamicConstants,
        cb: &CommandBuffer,
    ) -> RetiredRenderGraph {
        let device = params.device;
        let resources: Vec<AnyRenderResource> = self
            .rg
            .resources
            .iter()
            .zip(self.resource_info.image_usage_flags.iter())
            .map(
                |(resource, image_usage_flags): (&GraphResourceInfo, &vk::ImageUsageFlags)| {
                    match resource {
                        GraphResourceInfo::Created(resource) => match resource.desc {
                            GraphResourceDesc::Image(mut desc) => {
                                desc.usage = *image_usage_flags;

                                let image = transient_resource_cache
                                    .get_image(&desc)
                                    .unwrap_or_else(|| device.create_image(desc, None).unwrap());

                                AnyRenderResource::OwnedImage(image)
                            }
                        },
                        GraphResourceInfo::Imported(resource) => match resource {
                            GraphResourceImportInfo::Image(image) => {
                                AnyRenderResource::ImportedImage(image.clone())
                            }
                        },
                    }
                },
            )
            .collect();

        let mut resource_registry = ResourceRegistry {
            execution_params: &params,
            resources,
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
                    match resource.borrow() {
                        AnyRenderResourceRef::Image(image) => {
                            record_image_barrier(
                                &params.device.raw,
                                cb.raw,
                                ImageBarrier::new(
                                    image.raw,
                                    vk_sync::AccessType::Nothing, // TODO
                                    access.access_type,
                                    image_aspect_mask_from_access_type_and_format(
                                        access.access_type,
                                        image.desc.format,
                                    )
                                    .unwrap_or_else(|| {
                                        panic!(
                                            "Invalid image access {:?} :: {:?}",
                                            access.access_type, image.desc
                                        )
                                    }),
                                ),
                            );
                        }
                        AnyRenderResourceRef::Buffer(_) => {
                            todo!();
                        }
                    }
                }
            }

            let mut api = RenderPassApi {
                cb,
                resources: &mut resource_registry,
            };

            if let Some(render_fn) = pass.render_fn {
                render_fn(&mut api);
            }
        }

        RetiredRenderGraph {
            resources: resource_registry.resources,
        }
    }
}

pub struct RetiredRenderGraph {
    resources: Vec<AnyRenderResource>,
}

impl RetiredRenderGraph {
    pub fn get_image(&self, handle: ExportedHandle<Image>) -> &Image {
        Image::borrow_resource(&self.resources[handle.0.raw.id as usize])
    }

    pub fn release_resources(self, transient_resource_cache: &mut TransientResourceCache) {
        for resource in self.resources {
            match resource {
                AnyRenderResource::OwnedImage(image) => {
                    transient_resource_cache.insert_image(image)
                }
                AnyRenderResource::ImportedImage(_) => {}
                AnyRenderResource::OwnedBuffer(_) => todo!(),
            }
        }
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
