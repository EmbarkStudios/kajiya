#![allow(unused_imports)]

use super::{
    pass_builder::PassBuilder,
    resource::*,
    resource_registry::AnyRenderResourceRef,
    resource_registry::RegistryResource,
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
    backend::shader::PipelineShader,
    backend::{
        profiler::VkProfilerData,
        ray_tracing::{RayTracingAcceleration, RayTracingPipelineDesc},
        shader::RasterPipelineDesc,
    },
    dynamic_constants::DynamicConstants,
    pipeline_cache::ComputePipelineHandle,
    pipeline_cache::PipelineCache,
    pipeline_cache::{RasterPipelineHandle, RtPipelineHandle},
    transient_resource_cache::TransientResourceCache,
};
use ash::{version::DeviceV1_0, vk};
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
}

pub(crate) enum GraphResourceImportInfo {
    Image {
        resource: Arc<Image>,
        access_type: vk_sync::AccessType,
    },
    Buffer {
        resource: Arc<Buffer>,
        access_type: vk_sync::AccessType,
    },
    RayTracingAcceleration {
        resource: Arc<RayTracingAcceleration>,
        access_type: vk_sync::AccessType,
    },
}

pub(crate) enum GraphResourceInfo {
    Created(GraphResourceCreateInfo),
    Imported(GraphResourceImportInfo),
}

pub(crate) enum ExportableGraphResource {
    Image(Handle<Image>),
    Buffer(Handle<Buffer>),
}

impl ExportableGraphResource {
    fn raw(&self) -> GraphRawResourceHandle {
        match self {
            ExportableGraphResource::Image(h) => h.raw,
            ExportableGraphResource::Buffer(h) => h.raw,
        }
    }
}

#[derive(Clone, Copy)]
pub struct RgComputePipelineHandle {
    pub(crate) id: usize,
}

pub(crate) struct RgComputePipeline {
    pub(crate) shader_path: PathBuf,
    pub(crate) desc: ComputePipelineDesc,
}

#[derive(Clone, Copy)]
pub struct RgRasterPipelineHandle {
    pub(crate) id: usize,
}

pub(crate) struct RgRasterPipeline {
    pub(crate) shaders: Vec<PipelineShader<&'static str>>, // TODO, HACK
    pub(crate) desc: RasterPipelineDesc,
}

#[derive(Clone, Copy)]
pub struct RgRtPipelineHandle {
    pub(crate) id: usize,
}

pub(crate) struct RgRtPipeline {
    pub(crate) shaders: Vec<PipelineShader<&'static str>>, // TODO, HACK
    pub(crate) desc: RayTracingPipelineDesc,
}

pub struct PredefinedDescriptorSet {
    pub bindings: HashMap<u32, rspirv_reflect::DescriptorInfo>,
}

pub struct RenderGraph {
    passes: Vec<RecordedPass>,
    resources: Vec<GraphResourceInfo>,
    exported_resources: Vec<(ExportableGraphResource, vk_sync::AccessType)>,
    pub(crate) compute_pipelines: Vec<RgComputePipeline>,
    pub(crate) raster_pipelines: Vec<RgRasterPipeline>,
    pub(crate) rt_pipelines: Vec<RgRtPipeline>,
    pub predefined_descriptor_set_layouts: HashMap<u32, PredefinedDescriptorSet>,
}

pub trait ImportExportToRenderGraph
where
    Self: Resource + Sized,
{
    fn import(
        self: Arc<Self>,
        rg: &mut RenderGraph,
        access_type_at_import_time: vk_sync::AccessType,
    ) -> Handle<Self>;

    fn export(
        resource: Handle<Self>,
        rg: &mut RenderGraph,
        access_type: vk_sync::AccessType,
    ) -> ExportedHandle<Self>;
}

impl ImportExportToRenderGraph for Image {
    fn import(
        self: Arc<Self>,
        rg: &mut RenderGraph,
        access_type_at_import_time: vk_sync::AccessType,
    ) -> Handle<Self> {
        let res = GraphRawResourceHandle {
            id: rg.resources.len() as u32,
            version: 0,
        };

        let desc = self.desc;

        rg.resources.push(GraphResourceInfo::Imported(
            GraphResourceImportInfo::Image {
                resource: self,
                access_type: access_type_at_import_time,
            },
        ));

        Handle {
            raw: res,
            desc,
            marker: PhantomData,
        }
    }

    fn export(
        resource: Handle<Self>,
        rg: &mut RenderGraph,
        access_type: vk_sync::AccessType,
    ) -> ExportedHandle<Self> {
        let res = ExportedHandle {
            raw: resource.raw,
            marker: PhantomData,
        };
        rg.exported_resources
            .push((ExportableGraphResource::Image(resource), access_type));
        res
    }
}

impl ImportExportToRenderGraph for Buffer {
    fn import(
        self: Arc<Self>,
        rg: &mut RenderGraph,
        access_type_at_import_time: vk_sync::AccessType,
    ) -> Handle<Self> {
        let res = GraphRawResourceHandle {
            id: rg.resources.len() as u32,
            version: 0,
        };

        let desc = self.desc;

        rg.resources.push(GraphResourceInfo::Imported(
            GraphResourceImportInfo::Buffer {
                resource: self,
                access_type: access_type_at_import_time,
            },
        ));

        Handle {
            raw: res,
            desc,
            marker: PhantomData,
        }
    }

    fn export(
        resource: Handle<Self>,
        rg: &mut RenderGraph,
        access_type: vk_sync::AccessType,
    ) -> ExportedHandle<Self> {
        let res = ExportedHandle {
            raw: resource.raw,
            marker: PhantomData,
        };
        rg.exported_resources
            .push((ExportableGraphResource::Buffer(resource), access_type));
        res
    }
}

impl ImportExportToRenderGraph for RayTracingAcceleration {
    fn import(
        self: Arc<Self>,
        rg: &mut RenderGraph,
        access_type_at_import_time: vk_sync::AccessType,
    ) -> Handle<Self> {
        let res = GraphRawResourceHandle {
            id: rg.resources.len() as u32,
            version: 0,
        };

        let desc = RayTracingAccelerationDesc;

        rg.resources.push(GraphResourceInfo::Imported(
            GraphResourceImportInfo::RayTracingAcceleration {
                resource: self,
                access_type: access_type_at_import_time,
            },
        ));

        Handle {
            raw: res,
            desc,
            marker: PhantomData,
        }
    }

    fn export(
        _resource: Handle<Self>,
        _rg: &mut RenderGraph,
        _access_type: vk_sync::AccessType,
    ) -> ExportedHandle<Self> {
        todo!()
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

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resources: Vec::new(),
            exported_resources: Vec::new(),
            compute_pipelines: Vec::new(),
            raster_pipelines: Vec::new(),
            rt_pipelines: Vec::new(),
            predefined_descriptor_set_layouts: HashMap::new(),
        }
    }

    pub fn create<Desc: ResourceDesc>(
        &mut self,
        desc: Desc,
    ) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        let handle: Handle<<Desc as ResourceDesc>::Resource> = Handle {
            raw: self.create_raw_resource(GraphResourceCreateInfo {
                desc: desc.clone().into(),
            }),
            desc: TypeEquals::same(desc.clone()),
            marker: PhantomData,
        };

        handle
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

    pub fn import<Res: ImportExportToRenderGraph>(
        &mut self,
        resource: Arc<Res>,
        access_type_at_import_time: vk_sync::AccessType,
    ) -> Handle<Res> {
        ImportExportToRenderGraph::import(resource, self, access_type_at_import_time)
    }

    pub fn export<Res: ImportExportToRenderGraph>(
        &mut self,
        resource: Handle<Res>,
        access_type: vk_sync::AccessType,
    ) -> ExportedHandle<Res> {
        ImportExportToRenderGraph::export(resource, self, access_type)
    }

    /*pub fn export_image(
        &mut self,
        img: Handle<Image>,
        access_type: vk_sync::AccessType,
    ) -> ExportedHandle<Image> {
        self.exported_images.push((img.raw, access_type));
        ExportedHandle(img)
    }*/
}

#[derive(Debug)]
struct ResourceLifetime {
    first_access: Option<usize>,
    last_access: Option<usize>,
}

struct ResourceInfo {
    _lifetimes: Vec<ResourceLifetime>,
    image_usage_flags: Vec<vk::ImageUsageFlags>,
    buffer_usage_flags: Vec<vk::BufferUsageFlags>,
}

pub struct RenderGraphExecutionParams<'a> {
    pub device: &'a Device,
    pub pipeline_cache: &'a mut PipelineCache,
    pub frame_descriptor_set: vk::DescriptorSet,
    pub frame_constants_offset: u32,
    pub profiler_data: &'a VkProfilerData,
}

pub struct CompiledRenderGraph {
    rg: RenderGraph,
    resource_info: ResourceInfo,
    compute_pipelines: Vec<ComputePipelineHandle>,
    raster_pipelines: Vec<RasterPipelineHandle>,
    rt_pipelines: Vec<RtPipelineHandle>,
}

impl RenderGraph {
    pub fn add_pass<'s>(&'s mut self, name: &str) -> PassBuilder<'s> {
        let pass_idx = self.passes.len();

        PassBuilder {
            rg: self,
            pass_idx,
            pass: Some(RecordedPass::new(name)),
        }
    }

    fn calculate_resource_info(&self) -> ResourceInfo {
        let mut lifetimes: Vec<ResourceLifetime> = self
            .resources
            .iter()
            .map(|res| match res {
                GraphResourceInfo::Created(_) => ResourceLifetime {
                    first_access: None,
                    last_access: None,
                },
                GraphResourceInfo::Imported(_) => ResourceLifetime {
                    first_access: Some(0),
                    last_access: Some(0),
                },
            })
            .collect();

        let mut image_usage_flags: Vec<vk::ImageUsageFlags> =
            vec![Default::default(); self.resources.len()];

        let mut buffer_usage_flags: Vec<vk::BufferUsageFlags> =
            vec![Default::default(); self.resources.len()];

        for (res_idx, resource) in self.resources.iter().enumerate() {
            match resource {
                GraphResourceInfo::Created(GraphResourceCreateInfo {
                    desc: GraphResourceDesc::Image(desc),
                    ..
                }) => {
                    image_usage_flags[res_idx] = desc.usage;
                }
                GraphResourceInfo::Created(GraphResourceCreateInfo {
                    desc: GraphResourceDesc::Buffer(desc),
                    ..
                }) => {
                    buffer_usage_flags[res_idx] = desc.usage;
                }
                _ => {}
            }
        }

        for (pass_idx, pass) in self.passes.iter().enumerate() {
            for res_access in pass.read.iter().chain(pass.write.iter()) {
                let resource_index = res_access.handle.id as usize;
                let res = &mut lifetimes[resource_index];
                res.last_access = Some(
                    res.last_access
                        .map(|last_access| last_access.max(pass_idx))
                        .unwrap_or(pass_idx),
                );

                let access_mask = get_access_info(res_access.access.access_type).access_mask;

                match &self.resources[resource_index] {
                    // Images
                    GraphResourceInfo::Created(GraphResourceCreateInfo {
                        desc: GraphResourceDesc::Image(_),
                        ..
                    })
                    | GraphResourceInfo::Imported(GraphResourceImportInfo::Image { .. }) => {
                        let image_usage: vk::ImageUsageFlags =
                            image_access_mask_to_usage_flags(access_mask);

                        image_usage_flags[res_access.handle.id as usize] |= image_usage;
                    }

                    // Buffers
                    GraphResourceInfo::Created(GraphResourceCreateInfo {
                        desc: GraphResourceDesc::Buffer(_),
                        ..
                    })
                    | GraphResourceInfo::Imported(GraphResourceImportInfo::Buffer { .. }) => {
                        let buffer_usage: vk::BufferUsageFlags =
                            buffer_access_mask_to_usage_flags(access_mask);

                        buffer_usage_flags[res_access.handle.id as usize] |= buffer_usage;
                    }

                    // Acceleration structures
                    GraphResourceInfo::Created(GraphResourceCreateInfo {
                        desc: GraphResourceDesc::RayTracingAcceleration(_),
                        ..
                    }) => {
                        unimplemented!("Creation of acceleration structures via the render graph is not currently supported");
                    }
                    GraphResourceInfo::Imported(
                        GraphResourceImportInfo::RayTracingAcceleration { .. },
                    ) => {
                        // TODO; not currently tracking usage flags for RT acceleration
                    }
                };
            }
        }

        for (res, access_type) in &self.exported_resources {
            let raw_id = res.raw().id as usize;
            lifetimes[raw_id].last_access = Some(self.passes.len().saturating_sub(1));

            if *access_type != vk_sync::AccessType::Nothing {
                let access_mask = get_access_info(*access_type).access_mask;

                match res {
                    ExportableGraphResource::Image(_) => {
                        image_usage_flags[raw_id] |= image_access_mask_to_usage_flags(access_mask);
                    }
                    ExportableGraphResource::Buffer(_) => {
                        buffer_usage_flags[raw_id] |=
                            buffer_access_mask_to_usage_flags(access_mask);
                    }
                }
            }
        }

        ResourceInfo {
            _lifetimes: lifetimes,
            image_usage_flags,
            buffer_usage_flags,
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

        let raster_pipelines = self
            .raster_pipelines
            .iter()
            .map(|pipeline| pipeline_cache.register_raster(&pipeline.shaders, &pipeline.desc))
            .collect::<Vec<_>>();

        let rt_pipelines = self
            .rt_pipelines
            .iter()
            .map(|pipeline| pipeline_cache.register_ray_tracing(&pipeline.shaders, &pipeline.desc))
            .collect::<Vec<_>>();

        CompiledRenderGraph {
            rg: self,
            resource_info,
            compute_pipelines,
            raster_pipelines,
            rt_pipelines,
        }
    }

    pub(crate) fn record_pass(&mut self, pass: RecordedPass) {
        self.passes.push(pass);
    }
}

fn image_access_mask_to_usage_flags(access_mask: vk::AccessFlags) -> vk::ImageUsageFlags {
    match access_mask {
        vk::AccessFlags::SHADER_READ => vk::ImageUsageFlags::SAMPLED,
        vk::AccessFlags::SHADER_WRITE => vk::ImageUsageFlags::STORAGE,
        vk::AccessFlags::COLOR_ATTACHMENT_READ => vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::AccessFlags::COLOR_ATTACHMENT_WRITE => vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ => {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        }
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE => {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        }
        vk::AccessFlags::TRANSFER_READ => vk::ImageUsageFlags::TRANSFER_SRC,
        vk::AccessFlags::TRANSFER_WRITE => vk::ImageUsageFlags::TRANSFER_DST,
        _ if access_mask == vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE => {
            vk::ImageUsageFlags::STORAGE
        }
        _ => panic!("Invalid image access mask: {:?}", access_mask),
    }
}

fn buffer_access_mask_to_usage_flags(access_mask: vk::AccessFlags) -> vk::BufferUsageFlags {
    match access_mask {
        vk::AccessFlags::INDIRECT_COMMAND_READ => vk::BufferUsageFlags::INDIRECT_BUFFER,
        vk::AccessFlags::INDEX_READ => vk::BufferUsageFlags::INDEX_BUFFER,
        vk::AccessFlags::VERTEX_ATTRIBUTE_READ => vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
        vk::AccessFlags::UNIFORM_READ => vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::AccessFlags::SHADER_READ => vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
        vk::AccessFlags::SHADER_WRITE => vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::AccessFlags::TRANSFER_READ => vk::BufferUsageFlags::TRANSFER_SRC,
        vk::AccessFlags::TRANSFER_WRITE => vk::BufferUsageFlags::TRANSFER_DST,
        _ if access_mask == vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE => {
            vk::BufferUsageFlags::STORAGE_BUFFER
        }
        _ => panic!("Invalid buffer access mask: {:?}", access_mask),
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
        let resources: Vec<RegistryResource> = self
            .rg
            .resources
            .iter()
            .enumerate()
            .map(|(resource_idx, resource)| match resource {
                GraphResourceInfo::Created(resource) => match resource.desc {
                    GraphResourceDesc::Image(mut desc) => {
                        desc.usage = self.resource_info.image_usage_flags[resource_idx];

                        let image = transient_resource_cache
                            .get_image(&desc)
                            .unwrap_or_else(|| device.create_image(desc, None).unwrap());

                        RegistryResource {
                            access_type: vk_sync::AccessType::Nothing,
                            resource: AnyRenderResource::OwnedImage(image),
                        }
                    }
                    GraphResourceDesc::Buffer(mut desc) => {
                        desc.usage = self.resource_info.buffer_usage_flags[resource_idx];

                        let buffer = transient_resource_cache
                            .get_buffer(&desc)
                            .unwrap_or_else(|| device.create_buffer(desc, None).unwrap());

                        RegistryResource {
                            resource: AnyRenderResource::OwnedBuffer(buffer),
                            access_type: vk_sync::AccessType::Nothing,
                        }
                    }
                    GraphResourceDesc::RayTracingAcceleration(_) => {
                        unimplemented!();
                    }
                },
                GraphResourceInfo::Imported(resource) => match resource {
                    GraphResourceImportInfo::Image {
                        resource,
                        access_type,
                    } => RegistryResource {
                        resource: AnyRenderResource::ImportedImage(resource.clone()),
                        access_type: *access_type,
                    },
                    GraphResourceImportInfo::Buffer {
                        resource,
                        access_type,
                    } => RegistryResource {
                        resource: AnyRenderResource::ImportedBuffer(resource.clone()),
                        access_type: *access_type,
                    },
                    GraphResourceImportInfo::RayTracingAcceleration {
                        resource,
                        access_type,
                    } => RegistryResource {
                        resource: AnyRenderResource::ImportedRayTracingAcceleration(
                            resource.clone(),
                        ),
                        access_type: *access_type,
                    },
                },
            })
            .collect();

        let mut resource_registry = ResourceRegistry {
            execution_params: &params,
            resources,
            dynamic_constants: dynamic_constants,
            compute_pipelines: self.compute_pipelines,
            raster_pipelines: self.raster_pipelines,
            rt_pipelines: self.rt_pipelines,
        };

        for (pass_idx, pass) in self.rg.passes.into_iter().enumerate() {
            let vk_query_idx = {
                let query_id = crate::gpu_profiler::create_gpu_query(pass.name.as_str(), pass_idx);
                let vk_query_idx = params.profiler_data.get_query_id(query_id);

                unsafe {
                    params.device.raw.cmd_write_timestamp(
                        cb.raw,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        params.profiler_data.query_pool,
                        vk_query_idx * 2 + 0,
                    );
                }

                vk_query_idx
            };

            {
                let mut transitions: Vec<(usize, PassResourceAccessType)> = Vec::new();
                for resource_ref in pass.read.iter().chain(pass.write.iter()) {
                    transitions.push((resource_ref.handle.id as usize, resource_ref.access));
                }

                // TODO: optimize the barriers

                for (resource_idx, access) in transitions {
                    let resource = &mut resource_registry.resources[resource_idx];
                    Self::transition_resource(params.device, cb, resource, access.access_type);
                }
            }

            let mut api = RenderPassApi {
                cb,
                resources: &mut resource_registry,
            };

            if let Some(render_fn) = pass.render_fn {
                render_fn(&mut api);
            }

            unsafe {
                params.device.raw.cmd_write_timestamp(
                    cb.raw,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    params.profiler_data.query_pool,
                    vk_query_idx * 2 + 1,
                );
            }
        }

        // Transition exported images to the requested access types
        for (resource_idx, access_type) in self.rg.exported_resources {
            if access_type != vk_sync::AccessType::Nothing {
                let resource = &mut resource_registry.resources[resource_idx.raw().id as usize];
                Self::transition_resource(params.device, cb, resource, access_type);
            }
        }

        RetiredRenderGraph {
            resources: resource_registry.resources,
        }
    }

    fn transition_resource(
        device: &Device,
        cb: &CommandBuffer,
        resource: &mut RegistryResource,
        access_type: vk_sync::AccessType,
    ) {
        match resource.resource.borrow() {
            AnyRenderResourceRef::Image(image) => {
                record_image_barrier(
                    device,
                    cb.raw,
                    ImageBarrier::new(
                        image.raw,
                        resource.access_type,
                        access_type,
                        image_aspect_mask_from_access_type_and_format(
                            access_type,
                            image.desc.format,
                        )
                        .unwrap_or_else(|| {
                            panic!("Invalid image access {:?} :: {:?}", access_type, image.desc)
                        }),
                    ),
                );

                resource.access_type = access_type;
            }
            AnyRenderResourceRef::Buffer(_buffer) => {
                global_barrier(device, cb, &[resource.access_type], &[access_type]);

                resource.access_type = access_type;
            }
            AnyRenderResourceRef::RayTracingAcceleration(_) => {
                /*global_barrier(
                    device,
                    cb,
                    &[resource.access_type],
                    &[access_type],
                );*/
                // TODO

                resource.access_type = access_type;
            }
        }
    }
}

fn global_barrier(
    device: &Device,
    cb: &CommandBuffer,
    previous_accesses: &[vk_sync::AccessType],
    next_accesses: &[vk_sync::AccessType],
) {
    vk_sync::cmd::pipeline_barrier(
        device.raw.fp_v1_0(),
        cb.raw,
        Some(vk_sync::GlobalBarrier {
            previous_accesses,
            next_accesses,
        }),
        &[],
        &[],
    );
}

pub struct RetiredRenderGraph {
    resources: Vec<RegistryResource>,
}

impl RetiredRenderGraph {
    pub fn exported_resource<Res: Resource>(
        &self,
        handle: ExportedHandle<Res>,
    ) -> (&Res::Impl, vk_sync::AccessType) {
        let reg_resource = &self.resources[handle.raw.id as usize];
        (
            <Res as Resource>::borrow_resource(&reg_resource.resource),
            reg_resource.access_type,
        )
    }

    pub fn release_resources(self, transient_resource_cache: &mut TransientResourceCache) {
        for resource in self.resources {
            match resource.resource {
                AnyRenderResource::OwnedImage(image) => {
                    transient_resource_cache.insert_image(image)
                }
                AnyRenderResource::OwnedBuffer(buffer) => {
                    transient_resource_cache.insert_buffer(buffer)
                }
                AnyRenderResource::ImportedImage(_)
                | AnyRenderResource::ImportedBuffer(_)
                | AnyRenderResource::ImportedRayTracingAcceleration(_) => {}
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

pub(crate) struct RecordedPass {
    pub read: Vec<PassResourceRef>,
    pub write: Vec<PassResourceRef>,
    pub render_fn: Option<Box<DynRenderFn>>,
    pub name: String,
}

impl RecordedPass {
    fn new(name: &str) -> Self {
        Self {
            read: Default::default(),
            write: Default::default(),
            render_fn: Default::default(),
            name: name.to_owned(),
        }
    }
}
