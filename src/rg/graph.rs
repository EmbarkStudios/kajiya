#![allow(unused_imports)]

use super::{pass_builder::PassBuilder, resource::*, resource_registry::ResourceRegistry};

use crate::{
    backend::device::{CommandBuffer, Device},
    dynamic_constants::DynamicConstants,
    pipeline_cache::PipelineCache,
};
use ash::vk;
use std::{
    collections::HashMap,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
};

pub(crate) struct GraphResourceCreateInfo {
    pub desc: GraphResourceDesc,
    pub create_pass_idx: usize,
}

pub struct RenderGraph {
    passes: Vec<RecordedPass>,
    resources: Vec<GraphResourceCreateInfo>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resources: Vec::new(),
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

pub struct RenderGraphExecutionParams<'device, 'pipeline_cache> {
    pub device: &'device Device,
    pub pipeline_cache: &'pipeline_cache PipelineCache,
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

    fn calculate_resource_lifetimes(&self) -> Vec<ResourceLifetime> {
        let mut resource_lifetimes: Vec<ResourceLifetime> = self
            .resources
            .iter()
            .map(|res| ResourceLifetime {
                first_access: res.create_pass_idx,
                last_access: res.create_pass_idx,
            })
            .collect();

        for (pass_idx, pass) in self.passes.iter().enumerate() {
            for res_access in pass.read.iter().chain(pass.write.iter()) {
                let res = &mut resource_lifetimes[res_access.handle.id as usize];
                res.last_access = res.last_access.max(pass_idx);
            }
        }

        resource_lifetimes
    }

    pub fn execute<'device, 'pipeline_cache, 'cb, 'commands>(
        self,
        params: RenderGraphExecutionParams<'device, 'pipeline_cache>,
        dynamic_constants: &mut DynamicConstants,
        cb: &'cb mut CommandBuffer,
    ) -> anyhow::Result<()> {
        let _resource_lifetimes = self.calculate_resource_lifetimes();
        // TODO: alias resources

        /* println!(
            "Resources: {:#?}",
            self.resources
                .iter()
                .map(|info| info.desc)
                .zip(resource_lifetimes.iter())
                .collect::<Vec<_>>()
        ); */

        let device = params.device;

        let gpu_resources: Vec<RenderResourceHandle> = self
            .resources
            .iter()
            .map(|resource: &GraphResourceCreateInfo| match resource.desc {
                GraphResourceDesc::Image(desc) => {
                    let handle = handles.allocate_transient(RenderResourceType::Image);
                    device
                        .create_texture(
                            handle,
                            &RenderImageDesc {
                                texture_type: RenderImageType::Tex2d,
                                bind_flags: RenderBindFlags::UNORDERED_ACCESS
                                    | RenderBindFlags::SHADER_RESOURCE
                                    | RenderBindFlags::RENDER_TARGET,
                                format: desc.format,
                                width: desc.width,
                                height: desc.height,
                                depth: 1,
                                levels: 1,
                                elements: 1,
                            },
                            None,
                            "rg texture".into(),
                        )
                        .unwrap();

                    handle
                }
            })
            .collect();

        let mut resource_registry = ResourceRegistry {
            execution_params: &params,
            resources: gpu_resources,
            dynamic_constants: dynamic_constants,
        };

        let mut transitions = Vec::new();

        for pass in self.passes.into_iter() {
            transitions.clear();
            for resource_ref in pass.read.iter().chain(pass.write.iter()) {
                transitions.push((
                    resource_registry.resources[resource_ref.handle.id as usize],
                    resource_ref.access_mode,
                ));
            }
            cb.transitions(&transitions)?;

            (pass.render_fn.unwrap())(cb, &mut resource_registry)?;
        }

        Ok(())
    }

    pub(crate) fn record_pass(&mut self, pass: RecordedPass) {
        self.passes.push(pass);
    }
}

type DynRenderFn =
    dyn FnOnce(&mut RenderCommandList<'_>, &mut ResourceRegistry) -> anyhow::Result<()>;

enum ResourceUsageFlags {
    Image(vk::ImageUsageFlags),
    Buffer(vk::BufferUsageFlags),
}

pub(crate) struct PassResourceRef {
    pub handle: GraphRawResourceHandle,
    pub access_mode: ResourceUsageFlags,
}

#[derive(Default)]
pub(crate) struct RecordedPass {
    pub read: Vec<PassResourceRef>,
    pub write: Vec<PassResourceRef>,
    pub render_fn: Option<Box<DynRenderFn>>,
}
