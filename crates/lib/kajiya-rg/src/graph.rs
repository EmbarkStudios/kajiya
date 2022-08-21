#![allow(unused_imports)]

use crate::{renderer::FrameConstantsLayout, resource_registry::PendingRenderResourceInfo};

use super::{
    pass_builder::PassBuilder,
    resource::*,
    resource_registry::{
        AnyRenderResource, AnyRenderResourceRef, RegistryResource, ResourceRegistry,
    },
    RenderPassApi,
};

use kajiya_backend::{
    ash::{
        extensions::khr::Swapchain,
        vk::{self, DebugUtilsLabelEXT},
    },
    dynamic_constants::DynamicConstants,
    pipeline_cache::{
        ComputePipelineHandle, PipelineCache, RasterPipelineHandle, RtPipelineHandle,
    },
    rspirv_reflect,
    transient_resource_cache::TransientResourceCache,
    vk_sync,
    vulkan::{
        barrier::{
            get_access_info, image_aspect_mask_from_access_type_and_format, record_image_barrier,
            ImageBarrier,
        },
        device::{CommandBuffer, Device, VkProfilerData},
        image::ImageViewDesc,
        ray_tracing::{RayTracingAcceleration, RayTracingPipelineDesc},
        shader::{ComputePipelineDesc, PipelineShader, PipelineShaderDesc, RasterPipelineDesc},
    },
    BackendError,
};
use parking_lot::Mutex;
use std::{
    collections::{HashMap, VecDeque},
    ffi::CString,
    hash::Hash,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, Weak},
};

#[derive(Clone)]
pub(crate) struct GraphResourceCreateInfo {
    pub desc: GraphResourceDesc,
}

#[derive(Clone)]
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
    SwapchainImage,
}

#[derive(Clone)]
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
    pub(crate) desc: ComputePipelineDesc,
}

#[derive(Clone, Copy)]
pub struct RgRasterPipelineHandle {
    pub(crate) id: usize,
}

pub(crate) struct RgRasterPipeline {
    pub(crate) shaders: Vec<PipelineShaderDesc>,
    pub(crate) desc: RasterPipelineDesc,
}

#[derive(Clone, Copy)]
pub struct RgRtPipelineHandle {
    pub(crate) id: usize,
}

pub(crate) struct RgRtPipeline {
    pub(crate) shaders: Vec<PipelineShaderDesc>,
    pub(crate) desc: RayTracingPipelineDesc,
}

pub struct PredefinedDescriptorSet {
    pub bindings: HashMap<u32, rspirv_reflect::DescriptorInfo>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RenderDebugHook {
    pub name: String,
    pub id: u64,
}

#[derive(Clone, PartialEq, Eq)]
pub struct GraphDebugHook {
    pub render_debug_hook: RenderDebugHook,
}

pub struct RenderGraph {
    passes: Vec<RecordedPass>,
    resources: Vec<GraphResourceInfo>,
    exported_resources: Vec<(ExportableGraphResource, vk_sync::AccessType)>,
    pub(crate) compute_pipelines: Vec<RgComputePipeline>,
    pub(crate) raster_pipelines: Vec<RgRasterPipeline>,
    pub(crate) rt_pipelines: Vec<RgRtPipeline>,
    pub predefined_descriptor_set_layouts: HashMap<u32, PredefinedDescriptorSet>,

    pub debug_hook: Option<GraphDebugHook>,
    pub debugged_resource: Option<Handle<Image>>,
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

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
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
            debug_hook: None,
            debugged_resource: None,
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
            desc: TypeEquals::same(desc),
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

    pub fn get_swap_chain(&mut self) -> Handle<Image> {
        let res = GraphRawResourceHandle {
            id: self.resources.len() as u32,
            version: 0,
        };

        self.resources.push(GraphResourceInfo::Imported(
            GraphResourceImportInfo::SwapchainImage,
        ));

        Handle {
            raw: res,
            // TODO: size
            desc: ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, [1, 1]),
            marker: PhantomData,
        }
    }
}

#[derive(Debug)]
struct ResourceLifetime {
    //first_access: Option<usize>,
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
    pub frame_constants_layout: FrameConstantsLayout,
    pub profiler_data: &'a VkProfilerData,
}

pub struct RenderGraphPipelines {
    pub(crate) compute: Vec<ComputePipelineHandle>,
    pub(crate) raster: Vec<RasterPipelineHandle>,
    pub(crate) rt: Vec<RtPipelineHandle>,
}

pub struct CompiledRenderGraph {
    rg: RenderGraph,
    resource_info: ResourceInfo,
    pipelines: RenderGraphPipelines,
}

struct PendingDebugPass {
    img: Handle<Image>,
}

impl RenderGraph {
    pub fn add_pass<'s>(&'s mut self, name: &str) -> PassBuilder<'s> {
        let pass_idx = self.passes.len();

        PassBuilder {
            rg: self,
            pass_idx,
            pass: Some(RecordedPass::new(name, pass_idx)),
        }
    }

    fn calculate_resource_info(&self) -> ResourceInfo {
        let mut lifetimes: Vec<ResourceLifetime> = self
            .resources
            .iter()
            .map(|res| match res {
                GraphResourceInfo::Created(_) => ResourceLifetime {
                    //first_access: None,
                    last_access: None,
                },
                GraphResourceInfo::Imported(_) => ResourceLifetime {
                    //first_access: Some(0),
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
                    | GraphResourceInfo::Imported(GraphResourceImportInfo::Image { .. })
                    | GraphResourceInfo::Imported(GraphResourceImportInfo::SwapchainImage) => {
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
            .map(|pipeline| pipeline_cache.register_compute(&pipeline.desc))
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
            pipelines: RenderGraphPipelines {
                compute: compute_pipelines,
                raster: raster_pipelines,
                rt: rt_pipelines,
            },
        }
    }

    pub(crate) fn record_pass(&mut self, pass: RecordedPass) {
        let debug_pass = self.hook_debug_pass(&pass);
        self.passes.push(pass);

        if let Some(debug_pass) = debug_pass {
            let src_handle = debug_pass.img;
            let src_desc = *src_handle.desc();

            let mut dst = self.create(src_desc);
            let debug_pass = self.add_pass("debug");

            crate::SimpleRenderPass::new_compute(debug_pass, "/shaders/copy_color.hlsl")
                .read(&src_handle)
                .write(&mut dst)
                .dispatch(src_desc.extent);

            self.debugged_resource = Some(dst);
        }
    }

    fn hook_debug_pass(&mut self, pass: &RecordedPass) -> Option<PendingDebugPass> {
        let scope_hook = &self.debug_hook.as_ref()?.render_debug_hook;

        if pass.name == scope_hook.name && pass.idx as u64 == scope_hook.id {
            fn is_debug_compatible(desc: &ImageDesc) -> bool {
                kajiya_backend::vulkan::barrier::image_aspect_mask_from_format(desc.format)
                    == vk::ImageAspectFlags::COLOR
                    && desc.image_type == ImageType::Tex2d
            }

            // Grab the first compatible image written by this pass
            let (src_handle, src_desc) = pass.write.iter().find_map(|src_ref| {
                let src = &self.resources[src_ref.handle.id as usize];
                match src {
                    // Resources created by the render graph can be used as-is, as long as they have a color aspect
                    GraphResourceInfo::Created(GraphResourceCreateInfo {
                        desc: GraphResourceDesc::Image(img_desc),
                    }) if is_debug_compatible(img_desc) => Some((src_ref.handle, *img_desc)),

                    // Imported resources must also support vk::ImageUsageFlags::SAMPLED because their
                    // usage flags are supplied externally, and not derived by the graph
                    GraphResourceInfo::Imported(GraphResourceImportInfo::Image {
                        resource: img,
                        ..
                    }) if img.desc.usage.contains(vk::ImageUsageFlags::SAMPLED)
                        && is_debug_compatible(&img.desc) =>
                    {
                        Some((src_ref.handle, img.desc))
                    }
                    _ => None,
                }
            })?;

            let src_handle: Handle<Image> = Handle {
                raw: src_handle,
                desc: TypeEquals::same(src_desc)
                    .mip_levels(1)
                    .format(vk::Format::B10G11R11_UFLOAT_PACK32),
                marker: PhantomData,
            };

            Some(PendingDebugPass { img: src_handle })
        } else {
            None
        }
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

        // Appears with DepthAttachmentWriteStencilReadOnly
        _ if access_mask
            == vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE =>
        {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
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
    #[must_use]
    pub fn begin_execute<'exec_params, 'constants>(
        self,
        params: RenderGraphExecutionParams<'exec_params>,
        transient_resource_cache: &mut TransientResourceCache,
        dynamic_constants: &'constants mut DynamicConstants,
    ) -> ExecutingRenderGraph<'exec_params, 'constants> {
        let device = params.device;
        let resources: Vec<RegistryResource> = self
            .rg
            .resources
            .iter()
            .enumerate()
            .map(|(resource_idx, resource)| match resource {
                GraphResourceInfo::Created(create_info) => match create_info.desc {
                    GraphResourceDesc::Image(mut desc) => {
                        desc.usage = self.resource_info.image_usage_flags[resource_idx];

                        let image = transient_resource_cache
                            .get_image(&desc)
                            .unwrap_or_else(|| device.create_image(desc, vec![]).unwrap());

                        RegistryResource {
                            access_type: vk_sync::AccessType::Nothing,
                            resource: AnyRenderResource::OwnedImage(image),
                        }
                    }
                    GraphResourceDesc::Buffer(mut desc) => {
                        desc.usage = self.resource_info.buffer_usage_flags[resource_idx];

                        let buffer =
                            transient_resource_cache
                                .get_buffer(&desc)
                                .unwrap_or_else(|| {
                                    device.create_buffer(desc, "rg buffer", None).unwrap()
                                });

                        RegistryResource {
                            resource: AnyRenderResource::OwnedBuffer(buffer),
                            access_type: vk_sync::AccessType::Nothing,
                        }
                    }
                    GraphResourceDesc::RayTracingAcceleration(_) => {
                        unimplemented!();
                    }
                },
                GraphResourceInfo::Imported(import_info) => match import_info {
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
                    GraphResourceImportInfo::SwapchainImage => RegistryResource {
                        resource: AnyRenderResource::Pending(PendingRenderResourceInfo {
                            resource: resource.clone(),
                        }),
                        access_type: vk_sync::AccessType::ComputeShaderWrite,
                    },
                },
            })
            .collect();

        let resource_registry = ResourceRegistry {
            execution_params: params,
            resources,
            dynamic_constants,
            pipelines: self.pipelines,
        };

        ExecutingRenderGraph {
            resource_registry,
            passes: self.rg.passes.into(),
            resources: self.rg.resources,
            exported_resources: self.rg.exported_resources,
        }
    }
}

pub struct ExecutingRenderGraph<'exec_params, 'constants> {
    passes: VecDeque<RecordedPass>,
    resources: Vec<GraphResourceInfo>,
    exported_resources: Vec<(ExportableGraphResource, vk_sync::AccessType)>,
    resource_registry: ResourceRegistry<'exec_params, 'constants>,
}

impl<'exec_params, 'constants> ExecutingRenderGraph<'exec_params, 'constants> {
    pub fn record_main_cb(&mut self, cb: &CommandBuffer) {
        let mut first_presentation_pass: usize = self.passes.len();

        for (pass_idx, pass) in self.passes.iter().enumerate() {
            for res in &pass.write {
                let res = &self.resources[res.handle.id as usize];
                if matches!(
                    res,
                    GraphResourceInfo::Imported(GraphResourceImportInfo::SwapchainImage)
                ) {
                    first_presentation_pass = pass_idx;
                    break;
                }
            }
        }

        let mut passes: Vec<_> = std::mem::take(&mut self.passes).into();

        // At the start, transition all resources to the access type they're first used with
        // While we don't have split barriers yet, this will remove some bubbles
        // which would otherwise occur with temporal resources.
        {
            let mut resource_first_access_states: HashMap<u32, &mut PassResourceAccessType> =
                HashMap::with_capacity(self.resources.len());

            for pass in &mut passes[0..first_presentation_pass] {
                for resource_ref in pass.read.iter_mut().chain(pass.write.iter_mut()) {
                    resource_first_access_states
                        .entry(resource_ref.handle.id)
                        .or_insert(&mut resource_ref.access);
                }
            }

            let params = &self.resource_registry.execution_params;
            for (resource_idx, access) in resource_first_access_states {
                let resource = &mut self.resource_registry.resources[resource_idx as usize];
                Self::transition_resource(
                    params.device,
                    cb,
                    resource,
                    PassResourceAccessType {
                        access_type: access.access_type,
                        sync_type: PassResourceAccessSyncType::SkipSyncIfSameAccessType,
                    },
                    false,
                    "",
                );

                // Skip the sync when this pass is encountered later.
                access.sync_type = PassResourceAccessSyncType::SkipSyncIfSameAccessType;
            }
        }

        for pass in passes.drain(..first_presentation_pass) {
            Self::record_pass_cb(pass, &mut self.resource_registry, cb);
        }

        self.passes = passes.into();
    }

    #[must_use]
    pub fn record_presentation_cb(
        mut self,
        cb: &CommandBuffer,
        swapchain_image: Arc<Image>,
    ) -> RetiredRenderGraph {
        let params = &self.resource_registry.execution_params;

        // Transition exported images to the requested access types
        for (resource_idx, access_type) in self.exported_resources {
            if access_type != vk_sync::AccessType::Nothing {
                let resource =
                    &mut self.resource_registry.resources[resource_idx.raw().id as usize];
                Self::transition_resource(
                    params.device,
                    cb,
                    resource,
                    PassResourceAccessType {
                        access_type,
                        sync_type: PassResourceAccessSyncType::AlwaysSync,
                    },
                    false,
                    "",
                );
            }
        }

        for res in &mut self.resource_registry.resources {
            if let AnyRenderResource::Pending(pending) = &mut res.resource {
                match pending.resource {
                    GraphResourceInfo::Imported(GraphResourceImportInfo::SwapchainImage) => {
                        res.resource = AnyRenderResource::ImportedImage(swapchain_image.clone());
                    }
                    _ => panic!("Only swapchain can be currently pending"),
                }
            }
        }

        let passes = self.passes;
        for pass in passes {
            Self::record_pass_cb(pass, &mut self.resource_registry, cb);
        }

        RetiredRenderGraph {
            resources: self.resource_registry.resources,
        }
    }

    fn record_pass_cb(
        pass: RecordedPass,
        resource_registry: &mut ResourceRegistry,
        cb: &CommandBuffer,
    ) {
        let params = &resource_registry.execution_params;

        // Record a crash marker just before this pass
        params
            .device
            .record_crash_marker(cb, format!("begin render pass {:?}", pass.name));

        if let Some(debug_utils) = params.device.debug_utils() {
            unsafe {
                let label: CString = CString::new(pass.name.as_str()).unwrap();
                let label = DebugUtilsLabelEXT::builder().label_name(&label).build();
                debug_utils.cmd_begin_debug_utils_label(cb.raw, &label);
            }
        }

        let vk_scope = {
            let query_id = kajiya_backend::gpu_profiler::profiler().create_scope(&pass.name);
            params
                .profiler_data
                .begin_scope(&params.device.raw, cb.raw, query_id)
        };

        {
            let params = &resource_registry.execution_params;

            let mut transitions: Vec<(usize, PassResourceAccessType)> = Vec::new();
            for resource_ref in pass.read.iter() {
                transitions.push((
                    resource_ref.handle.id as usize,
                    resource_ref.access,
                    //format!("read {i}"),
                ));
            }

            for resource_ref in pass.write.iter() {
                transitions.push((
                    resource_ref.handle.id as usize,
                    resource_ref.access,
                    //format!("write {i}"),
                ));
            }

            // TODO: optimize the barriers

            for (resource_idx, access) in transitions {
                let resource = &mut resource_registry.resources[resource_idx];

                Self::transition_resource(
                    params.device,
                    cb,
                    resource,
                    access,
                    //pass.name == "raster simple",
                    false,
                    "",
                );
            }
        }

        let mut api = RenderPassApi {
            cb,
            resources: resource_registry,
        };

        if let Some(render_fn) = pass.render_fn {
            if let Err(err) = render_fn(&mut api) {
                panic!("Pass {:?} failed to render: {:#}", pass.name, err);
            }
        }

        let params = &resource_registry.execution_params;

        params
            .profiler_data
            .end_scope(&params.device.raw, cb.raw, vk_scope);

        if let Some(debug_utils) = params.device.debug_utils() {
            unsafe {
                debug_utils.cmd_end_debug_utils_label(cb.raw);
            }
        }

        // Record a crash marker just after this pass
        params
            .device
            .record_crash_marker(cb, format!("end render pass {:?}", pass.name));
    }

    fn transition_resource(
        device: &Device,
        cb: &CommandBuffer,
        resource: &mut RegistryResource,
        access: PassResourceAccessType,
        debug: bool,
        dbg_str: &str,
    ) {
        if unsafe { RG_ALLOW_PASS_OVERLAP }
            && resource.access_type == access.access_type
            && matches!(
                access.sync_type,
                PassResourceAccessSyncType::SkipSyncIfSameAccessType
            )
        {
            return;
        }

        if debug {
            log::info!(
                "\t{dbg_str}: {:?} -> {:?}",
                resource.access_type,
                access.access_type
            );
        }

        match resource.resource.borrow() {
            AnyRenderResourceRef::Image(image) => {
                if debug {
                    log::info!("\t(image {:?})", image.desc);
                }

                record_image_barrier(
                    device,
                    cb.raw,
                    ImageBarrier::new(
                        image.raw,
                        resource.access_type,
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

                resource.access_type = access.access_type;
            }
            AnyRenderResourceRef::Buffer(buffer) => {
                if debug {
                    log::info!("\t(buffer {:?})", buffer.desc);
                }
                //global_barrier(device, cb, &[resource.access_type], &[access.access_type]);

                vk_sync::cmd::pipeline_barrier(
                    device.raw.fp_v1_0(),
                    cb.raw,
                    None,
                    &[vk_sync::BufferBarrier {
                        previous_accesses: &[resource.access_type],
                        next_accesses: &[access.access_type],
                        src_queue_family_index: device.universal_queue.family.index,
                        dst_queue_family_index: device.universal_queue.family.index,
                        buffer: buffer.raw,
                        offset: 0,
                        size: buffer.desc.size,
                    }],
                    &[],
                );

                resource.access_type = access.access_type;
            }
            AnyRenderResourceRef::RayTracingAcceleration(_) => {
                if debug {
                    log::info!("\t(bvh)");
                }
                /*global_barrier(
                    device,
                    cb,
                    &[resource.access_type],
                    &[access_type],
                );*/
                // TODO

                resource.access_type = access.access_type;
            }
        }
    }
}

#[allow(dead_code)]
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
    ) -> (&Res, vk_sync::AccessType) {
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
                | AnyRenderResource::ImportedRayTracingAcceleration(_) => {},
                AnyRenderResource::Pending { .. } => panic!("RetiredRenderGraph::release_resources called while a resource was in Pending state"),
            }
        }
    }
}

type DynRenderFn = dyn FnOnce(&mut RenderPassApi) -> Result<(), BackendError>;

#[derive(Copy, Clone)]
pub enum PassResourceAccessSyncType {
    AlwaysSync,
    SkipSyncIfSameAccessType,
}

#[derive(Copy, Clone)]
pub struct PassResourceAccessType {
    // TODO: multiple
    access_type: vk_sync::AccessType,
    sync_type: PassResourceAccessSyncType,
}

impl PassResourceAccessType {
    pub fn new(access_type: vk_sync::AccessType, sync_type: PassResourceAccessSyncType) -> Self {
        Self {
            access_type,
            sync_type,
        }
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
    pub idx: usize,
}

impl RecordedPass {
    fn new(name: &str, idx: usize) -> Self {
        Self {
            read: Default::default(),
            write: Default::default(),
            render_fn: Default::default(),
            name: name.to_owned(),
            idx,
        }
    }
}

pub static mut RG_ALLOW_PASS_OVERLAP: bool = true;
