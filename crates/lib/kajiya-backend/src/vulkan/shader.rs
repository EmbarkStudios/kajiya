#![allow(dead_code)]

use super::{
    device::{Device, SamplerDesc},
    image::ImageDesc,
};
use crate::{chunky_list::TempList, shader_compiler::get_cs_local_size_from_spirv};
use arrayvec::ArrayVec;
use ash::vk;
use byte_slice_cast::AsSliceOf as _;
use bytes::Bytes;
use derive_builder::Builder;
use parking_lot::Mutex;
use std::{
    collections::{hash_map::Entry, HashMap},
    ffi::CString,
    path::PathBuf,
    sync::Arc,
};

pub const MAX_DESCRIPTOR_SETS: usize = 4;

type DescriptorSetLayout = HashMap<u32, rspirv_reflect::DescriptorInfo>;
type StageDescriptorSetLayouts = HashMap<u32, DescriptorSetLayout>;

pub struct ShaderPipelineCommon {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub set_layout_info: Vec<HashMap<u32, vk::DescriptorType>>,
    pub descriptor_pool_sizes: Vec<vk::DescriptorPoolSize>,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub pipeline_bind_point: vk::PipelineBindPoint,
}
pub struct ComputePipeline {
    pub common: ShaderPipelineCommon,
    pub group_size: [u32; 3],
}

impl std::ops::Deref for ComputePipeline {
    type Target = ShaderPipelineCommon;

    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

pub struct RasterPipeline {
    pub common: ShaderPipelineCommon,
}

impl std::ops::Deref for RasterPipeline {
    type Target = ShaderPipelineCommon;

    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

pub fn create_descriptor_set_layouts(
    device: &Device,
    descriptor_sets: &StageDescriptorSetLayouts,
    stage_flags: vk::ShaderStageFlags,
    set_opts: &[Option<(u32, DescriptorSetLayoutOpts)>; MAX_DESCRIPTOR_SETS],
) -> (
    Vec<vk::DescriptorSetLayout>,
    Vec<HashMap<u32, vk::DescriptorType>>,
) {
    // dbg!(&descriptor_sets);

    // Make a vector of Option<ref> to the original entries
    let mut set_opts = set_opts
        .iter()
        .map(|item| item.as_ref())
        .collect::<Vec<_>>();

    let samplers = TempList::new();

    // Find the number of sets in `descriptor_sets`
    let set_count = descriptor_sets
        .iter()
        .map(|(set_index, _)| *set_index + 1)
        .max()
        .unwrap_or(0u32);

    // Max that with the highest set in `set_opts`
    let set_count = set_count.max(
        set_opts
            .iter()
            .filter_map(|opt| opt.as_ref())
            .map(|(set_index, _)| *set_index + 1)
            .max()
            .unwrap_or(0u32),
    );

    let mut set_layouts: Vec<vk::DescriptorSetLayout> = Vec::with_capacity(set_count as usize);
    let mut set_layout_info: Vec<HashMap<u32, vk::DescriptorType>> =
        Vec::with_capacity(set_count as usize);

    for set_index in 0..set_count {
        let stage_flags = if 0 == set_index {
            stage_flags
        } else {
            // Set 0 is for draw params,
            // Further sets are for pass/frame bindings, and use all stage flags
            // TODO: pass those as a parameter here?
            vk::ShaderStageFlags::ALL
        };

        let _set_opts_default = Default::default();
        // Find the descriptor set opts corresponding to the set index, and remove them from the opts list
        let set_opts = {
            let mut resolved_set_opts: &DescriptorSetLayoutOpts = &_set_opts_default;

            for maybe_opt in set_opts.iter_mut() {
                if let Some(opt) = maybe_opt.as_mut() {
                    if opt.0 == set_index {
                        resolved_set_opts = &std::mem::take(maybe_opt).unwrap().1;
                    }
                }
            }
            resolved_set_opts
        };

        // Use the specified override, or the layout parsed from the shader if no override was provided
        let set = set_opts
            .replace
            .as_ref()
            .or_else(|| descriptor_sets.get(&set_index));

        if let Some(set) = set {
            let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::with_capacity(set.len());
            let mut binding_flags: Vec<vk::DescriptorBindingFlags> =
                vec![vk::DescriptorBindingFlags::PARTIALLY_BOUND; set.len()];

            let mut set_layout_create_flags = vk::DescriptorSetLayoutCreateFlags::empty();

            for (binding_index, binding) in set.iter() {
                match binding.ty {
                    rspirv_reflect::DescriptorType::UNIFORM_BUFFER
                    | rspirv_reflect::DescriptorType::UNIFORM_TEXEL_BUFFER
                    | rspirv_reflect::DescriptorType::STORAGE_IMAGE
                    | rspirv_reflect::DescriptorType::STORAGE_BUFFER
                    | rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => bindings.push(
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(*binding_index)
                            //.descriptor_count(binding.count)
                            .descriptor_count(1) // TODO
                            .descriptor_type(match binding.ty {
                                rspirv_reflect::DescriptorType::UNIFORM_BUFFER => {
                                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                                }
                                rspirv_reflect::DescriptorType::UNIFORM_TEXEL_BUFFER => {
                                    vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                                }
                                rspirv_reflect::DescriptorType::STORAGE_IMAGE => {
                                    vk::DescriptorType::STORAGE_IMAGE
                                }
                                rspirv_reflect::DescriptorType::STORAGE_BUFFER => {
                                    if binding.name.ends_with("_dyn") {
                                        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                                    } else {
                                        vk::DescriptorType::STORAGE_BUFFER
                                    }
                                }
                                rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                                    vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                                }
                                _ => unimplemented!("{:?}", binding),
                            })
                            .stage_flags(stage_flags)
                            .build(),
                    ),
                    rspirv_reflect::DescriptorType::SAMPLED_IMAGE => {
                        if matches!(
                            binding.dimensionality,
                            rspirv_reflect::DescriptorDimensionality::RuntimeArray
                        ) {
                            // Bindless

                            binding_flags[bindings.len()] =
                                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                                    | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
                                    | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                    | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;

                            set_layout_create_flags |=
                                vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
                        }

                        let descriptor_count = match binding.dimensionality {
                            rspirv_reflect::DescriptorDimensionality::Single => 1,
                            rspirv_reflect::DescriptorDimensionality::Array(size) => size,
                            rspirv_reflect::DescriptorDimensionality::RuntimeArray => {
                                device.max_bindless_descriptor_count()
                            }
                        };

                        bindings.push(
                            vk::DescriptorSetLayoutBinding::builder()
                                .binding(*binding_index)
                                .descriptor_count(descriptor_count) // TODO
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .stage_flags(stage_flags)
                                .build(),
                        );
                    }
                    rspirv_reflect::DescriptorType::SAMPLER => {
                        let name_prefix = "sampler_";
                        if let Some(mut spec) = binding.name.strip_prefix(name_prefix) {
                            let texel_filter = match &spec[..1] {
                                "n" => vk::Filter::NEAREST,
                                "l" => vk::Filter::LINEAR,
                                _ => panic!("{}", &spec[..1]),
                            };
                            spec = &spec[1..];

                            let mipmap_mode = match &spec[..1] {
                                "n" => vk::SamplerMipmapMode::NEAREST,
                                "l" => vk::SamplerMipmapMode::LINEAR,
                                _ => panic!("{}", &spec[..1]),
                            };
                            spec = &spec[1..];

                            let address_modes = match spec {
                                "r" => vk::SamplerAddressMode::REPEAT,
                                "mr" => vk::SamplerAddressMode::MIRRORED_REPEAT,
                                "c" => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                                "cb" => vk::SamplerAddressMode::CLAMP_TO_BORDER,
                                _ => panic!("{}", spec),
                            };

                            bindings.push(
                                vk::DescriptorSetLayoutBinding::builder()
                                    .descriptor_count(1)
                                    .descriptor_type(vk::DescriptorType::SAMPLER)
                                    .stage_flags(stage_flags)
                                    .binding(*binding_index)
                                    .immutable_samplers(std::slice::from_ref(samplers.add(
                                        device.get_sampler(SamplerDesc {
                                            texel_filter,
                                            mipmap_mode,
                                            address_modes,
                                        }),
                                    )))
                                    .build(),
                            );
                        } else {
                            panic!("{}", binding.name);
                        }
                    }
                    rspirv_reflect::DescriptorType::ACCELERATION_STRUCTURE_KHR => bindings.push(
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(*binding_index)
                            .descriptor_count(1) // TODO
                            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                            .stage_flags(stage_flags)
                            .build(),
                    ),

                    _ => unimplemented!("{:?}", binding),
                }
            }

            let mut binding_flags_create_info =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                    .binding_flags(&binding_flags);

            let set_layout = unsafe {
                device
                    .raw
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::builder()
                            .flags(set_opts.flags.unwrap_or_default() | set_layout_create_flags)
                            .bindings(&bindings)
                            .push_next(&mut binding_flags_create_info)
                            .build(),
                        None,
                    )
                    .unwrap()
            };

            set_layouts.push(set_layout);
            set_layout_info.push(
                bindings
                    .iter()
                    .map(|binding| (binding.binding, binding.descriptor_type))
                    .collect(),
            );
        } else {
            let set_layout = unsafe {
                device
                    .raw
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::builder().build(),
                        None,
                    )
                    .unwrap()
            };

            set_layouts.push(set_layout);
            set_layout_info.push(Default::default());
        }
    }

    (set_layouts, set_layout_info)
}

#[derive(Builder, Default, Debug, Clone)]
#[builder(pattern = "owned", derive(Clone))]
pub struct DescriptorSetLayoutOpts {
    #[builder(setter(strip_option), default)]
    pub flags: Option<vk::DescriptorSetLayoutCreateFlags>,
    #[builder(setter(strip_option), default)]
    pub replace: Option<DescriptorSetLayout>,
}

impl DescriptorSetLayoutOpts {
    pub fn builder() -> DescriptorSetLayoutOptsBuilder {
        DescriptorSetLayoutOptsBuilder::default()
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum ShaderSource {
    Rust { entry: String },
    Hlsl { path: PathBuf },
}

impl ShaderSource {
    pub fn rust(entry: impl Into<String>) -> Self {
        ShaderSource::Rust {
            entry: entry.into(),
        }
    }

    pub fn hlsl(path: impl Into<PathBuf>) -> Self {
        ShaderSource::Hlsl { path: path.into() }
    }

    pub fn entry(&self) -> &str {
        match self {
            ShaderSource::Rust { entry } => entry,
            ShaderSource::Hlsl { .. } => "main",
        }
    }
}

#[derive(Builder, Clone)]
#[builder(pattern = "owned", derive(Clone))]
pub struct ComputePipelineDesc {
    #[builder(default, setter(name = "descriptor_set_opts_impl"))]
    pub descriptor_set_opts: [Option<(u32, DescriptorSetLayoutOpts)>; MAX_DESCRIPTOR_SETS],
    #[builder(default)]
    pub push_constants_bytes: usize,
    pub source: ShaderSource,
}

impl ComputePipelineDescBuilder {
    pub fn descriptor_set_opts(mut self, opts: &[(u32, DescriptorSetLayoutOptsBuilder)]) -> Self {
        assert!(opts.len() <= MAX_DESCRIPTOR_SETS);
        let mut descriptor_set_opts: [Option<(u32, DescriptorSetLayoutOpts)>; MAX_DESCRIPTOR_SETS] =
            Default::default();
        for (i, (opt_set, opt)) in opts.iter().cloned().enumerate() {
            descriptor_set_opts[i] = Some((opt_set, opt.build().unwrap()));
        }
        self.descriptor_set_opts = Some(descriptor_set_opts);
        self
    }

    pub fn compute_rust(mut self, entry: impl Into<String>) -> Self {
        self.source = Some(ShaderSource::rust(entry));
        self
    }

    pub fn compute_hlsl(mut self, path: impl Into<PathBuf>) -> Self {
        self.source = Some(ShaderSource::hlsl(path));
        self
    }
}

impl ComputePipelineDesc {
    pub fn builder() -> ComputePipelineDescBuilder {
        ComputePipelineDescBuilder::default()
    }
}

pub fn create_compute_pipeline(
    device: &Device,
    spirv: &[u8],
    desc: &ComputePipelineDesc,
) -> ComputePipeline {
    let (descriptor_set_layouts, set_layout_info) = super::shader::create_descriptor_set_layouts(
        device,
        &rspirv_reflect::Reflection::new_from_spirv(spirv)
            .unwrap()
            .get_descriptor_sets()
            .unwrap(),
        vk::ShaderStageFlags::COMPUTE,
        &desc.descriptor_set_opts,
    );

    // dbg!(&set_layout_info);

    let mut layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

    let push_constant_ranges = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: desc.push_constants_bytes as _,
    };

    if desc.push_constants_bytes > 0 {
        layout_create_info =
            layout_create_info.push_constant_ranges(std::slice::from_ref(&push_constant_ranges));
    }

    unsafe {
        let shader_module = device
            .raw
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::builder().code(spirv.as_slice_of::<u32>().unwrap()),
                None,
            )
            .unwrap();

        let entry_name = CString::new(desc.source.entry()).unwrap();
        let stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(&entry_name);

        let pipeline_layout = device
            .raw
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_create_info.build())
            .layout(pipeline_layout);

        let pipeline = device
            .raw
            // TODO: pipeline cache
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
            .expect("pipeline")[0];

        let mut descriptor_pool_sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for bindings in set_layout_info.iter() {
            for ty in bindings.values() {
                if let Some(mut dps) = descriptor_pool_sizes.iter_mut().find(|item| item.ty == *ty)
                {
                    dps.descriptor_count += 1;
                } else {
                    descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                        ty: *ty,
                        descriptor_count: 1,
                    })
                }
            }
        }

        ComputePipeline {
            common: ShaderPipelineCommon {
                pipeline_layout,
                pipeline,
                set_layout_info,
                descriptor_pool_sizes,
                descriptor_set_layouts,
                pipeline_bind_point: vk::PipelineBindPoint::COMPUTE,
            },
            group_size: get_cs_local_size_from_spirv(spirv.as_slice_of::<u32>().unwrap()).unwrap(),
        }
    }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub enum ShaderPipelineStage {
    Vertex,
    Pixel,
    RayGen,
    RayMiss,
    RayClosestHit,
}

#[derive(Builder, Hash, PartialEq, Eq, Clone, Debug)]
#[builder(pattern = "owned")]
pub struct PipelineShaderDesc {
    pub stage: ShaderPipelineStage,
    #[builder(setter(strip_option), default)]
    pub descriptor_set_layout_flags: Option<Vec<(usize, vk::DescriptorSetLayoutCreateFlags)>>,
    #[builder(default)]
    pub push_constants_bytes: usize,
    #[builder(default = "\"main\".to_owned()")]
    pub entry: String,
    pub source: ShaderSource,
}

impl PipelineShaderDesc {
    pub fn builder(stage: ShaderPipelineStage) -> PipelineShaderDescBuilder {
        PipelineShaderDescBuilder::default().stage(stage)
    }
}

impl PipelineShaderDescBuilder {
    pub fn hlsl_source(mut self, path: impl Into<PathBuf>) -> Self {
        self.source = Some(ShaderSource::hlsl(path));

        self
    }

    pub fn rust_source(mut self, entry: impl Into<String>) -> Self {
        self.source = Some(ShaderSource::rust(entry));

        self
    }
}

#[derive(Builder, Clone)]
#[builder(pattern = "owned", derive(Clone))]
pub struct RasterPipelineDesc {
    #[builder(default)]
    pub descriptor_set_opts: [Option<(u32, DescriptorSetLayoutOpts)>; MAX_DESCRIPTOR_SETS],
    pub render_pass: Arc<RenderPass>,
    #[builder(default)]
    pub face_cull: bool,
    #[builder(default = "true")]
    pub depth_write: bool,
    #[builder(default)]
    pub push_constants_bytes: usize,
}

impl RasterPipelineDesc {
    pub fn builder() -> RasterPipelineDescBuilder {
        RasterPipelineDescBuilder::default()
    }
}

/*pub struct RasterPipeline {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub set_layout_info: Vec<HashMap<u32, vk::DescriptorType>>,
    pub descriptor_pool_sizes: Vec<vk::DescriptorPoolSize>,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    //pub render_pass: Arc<RenderPass>,
}*/

#[derive(Clone, Copy)]
pub struct RenderPassAttachmentDesc {
    pub format: vk::Format,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub samples: vk::SampleCountFlags,
}

impl RenderPassAttachmentDesc {
    pub fn new(format: vk::Format) -> Self {
        Self {
            format,
            load_op: vk::AttachmentLoadOp::LOAD,
            store_op: vk::AttachmentStoreOp::STORE,
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    pub fn garbage_input(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::DONT_CARE;
        self
    }

    pub fn clear_input(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self
    }

    pub fn discard_output(mut self) -> Self {
        self.store_op = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    fn to_vk(
        self,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> vk::AttachmentDescription {
        vk::AttachmentDescription {
            format: self.format,
            samples: self.samples,
            load_op: self.load_op,
            store_op: self.store_op,
            initial_layout,
            final_layout,
            ..Default::default()
        }
    }
}

pub const MAX_COLOR_ATTACHMENTS: usize = 8;

#[derive(Eq, PartialEq, Hash)]
pub struct FramebufferCacheKey {
    pub dims: [u32; 2],
    pub attachments:
        ArrayVec<[(vk::ImageUsageFlags, vk::ImageCreateFlags); MAX_COLOR_ATTACHMENTS + 1]>,
}

impl FramebufferCacheKey {
    pub fn new<'a>(
        dims: [u32; 2],
        color_attachments: impl Iterator<Item = &'a ImageDesc>,
        depth_stencil_attachment: Option<&'a ImageDesc>,
    ) -> Self {
        let color_attachments = color_attachments
            .chain(depth_stencil_attachment.into_iter())
            .copied()
            .map(|attachment| (attachment.usage, attachment.flags))
            .collect();

        Self {
            dims,
            attachments: color_attachments,
        }
    }
}

// TODO: nuke when resizing
pub struct FramebufferCache {
    entries: Mutex<HashMap<FramebufferCacheKey, vk::Framebuffer>>,
    attachment_desc: ArrayVec<[RenderPassAttachmentDesc; MAX_COLOR_ATTACHMENTS + 1]>,
    render_pass: vk::RenderPass,
    color_attachment_count: usize,
}

impl FramebufferCache {
    fn new(
        render_pass: vk::RenderPass,
        color_attachments: &[RenderPassAttachmentDesc],
        depth_attachment: Option<RenderPassAttachmentDesc>,
    ) -> Self {
        let mut attachment_desc = ArrayVec::new();

        attachment_desc
            .try_extend_from_slice(color_attachments)
            .unwrap();

        if let Some(depth_attachment) = depth_attachment {
            attachment_desc.push(depth_attachment)
        }

        Self {
            entries: Default::default(),
            attachment_desc,
            render_pass,
            color_attachment_count: color_attachments.len(),
        }
    }

    pub fn get_or_create(
        &self,
        device: &ash::Device,
        key: FramebufferCacheKey,
    ) -> anyhow::Result<vk::Framebuffer> {
        let mut entries = self.entries.lock();

        if let Some(entry) = entries.get(&key) {
            Ok(*entry)
        } else {
            let entry = {
                let color_formats = TempList::new();
                let [width, height] = key.dims;

                let attachments = self
                    .attachment_desc
                    .iter()
                    .zip(key.attachments.iter())
                    .map(|(desc, (usage, flags))| {
                        vk::FramebufferAttachmentImageInfoKHR::builder()
                            .width(width as _)
                            .height(height as _)
                            .flags(*flags)
                            .layer_count(1)
                            .view_formats(std::slice::from_ref(color_formats.add(desc.format)))
                            .usage(*usage)
                            .build()
                    })
                    .collect::<ArrayVec<[_; MAX_COLOR_ATTACHMENTS + 1]>>();

                let mut imageless_desc = vk::FramebufferAttachmentsCreateInfoKHR::builder()
                    .attachment_image_infos(&attachments);

                let mut fbo_desc = vk::FramebufferCreateInfo::builder()
                    .flags(vk::FramebufferCreateFlags::IMAGELESS_KHR)
                    .render_pass(self.render_pass)
                    .width(width as _)
                    .height(height as _)
                    .layers(1)
                    .push_next(&mut imageless_desc);

                fbo_desc.attachment_count = attachments.len() as _;

                unsafe { device.create_framebuffer(&fbo_desc, None)? }
            };

            entries.insert(key, entry);
            Ok(entry)
        }
    }
}

pub struct RenderPassDesc<'a> {
    pub color_attachments: &'a [RenderPassAttachmentDesc],
    pub depth_attachment: Option<RenderPassAttachmentDesc>,
}

pub struct RenderPass {
    pub raw: vk::RenderPass,
    pub framebuffer_cache: FramebufferCache,
}

pub fn create_render_pass(device: &Device, desc: RenderPassDesc<'_>) -> Arc<RenderPass> {
    let renderpass_attachments = desc
        .color_attachments
        .iter()
        .map(|a| {
            a.to_vk(
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            )
        })
        .chain(desc.depth_attachment.as_ref().map(|a| {
            a.to_vk(
                vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
                vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
            )
        }))
        .collect::<Vec<_>>();

    let color_attachment_refs = (0..desc.color_attachments.len() as u32)
        .map(|attachment| vk::AttachmentReference {
            attachment,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        })
        .collect::<Vec<_>>();

    let depth_attachment_ref = vk::AttachmentReference {
        attachment: desc.color_attachments.len() as u32,
        layout: vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
    };

    // TODO: Calculate optimal dependencies. using implicit dependencies for now.
    /*let dependencies = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ..Default::default()
    }];*/

    let mut subpass_description = vk::SubpassDescription::builder()
        .color_attachments(&color_attachment_refs)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

    if desc.depth_attachment.is_some() {
        subpass_description = subpass_description.depth_stencil_attachment(&depth_attachment_ref);
    }
    let subpass_description = subpass_description.build();

    let subpasses = [subpass_description];
    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&renderpass_attachments)
        .subpasses(&subpasses);

    let render_pass = unsafe {
        device
            .raw
            .create_render_pass(&render_pass_create_info, None)
            .unwrap()
    };

    Arc::new(RenderPass {
        raw: render_pass,
        framebuffer_cache: FramebufferCache::new(
            render_pass,
            desc.color_attachments,
            desc.depth_attachment,
        ),
    })
}

#[derive(Hash, PartialEq, Eq)]
pub struct PipelineShader<ShaderCode> {
    pub code: ShaderCode,
    pub desc: PipelineShaderDesc,
}

impl<ShaderCode> Clone for PipelineShader<ShaderCode>
where
    ShaderCode: Clone,
{
    fn clone(&self) -> Self {
        Self {
            code: self.code.clone(),
            desc: self.desc.clone(),
        }
    }
}
//impl<ShaderCode> Hash for RasterPipelineShader<ShaderCode> {}

impl<ShaderCode> PipelineShader<ShaderCode> {
    pub fn new(code: ShaderCode, desc: PipelineShaderDescBuilder) -> Self {
        Self {
            code,
            desc: desc.build().unwrap(),
        }
    }
}

pub fn create_raster_pipeline(
    device: &Device,
    shaders: &[PipelineShader<Bytes>],
    desc: &RasterPipelineDesc,
) -> anyhow::Result<RasterPipeline> {
    let stage_layouts = shaders
        .iter()
        .map(|shader| {
            rspirv_reflect::Reflection::new_from_spirv(&shader.code)
                .unwrap()
                .get_descriptor_sets()
                .unwrap()
        })
        .collect::<Vec<_>>();

    let (descriptor_set_layouts, set_layout_info) = super::shader::create_descriptor_set_layouts(
        device,
        &merge_shader_stage_layouts(stage_layouts),
        vk::ShaderStageFlags::ALL_GRAPHICS,
        //desc.descriptor_set_layout_flags.unwrap_or(&[]),  // TODO: merge flags
        &desc.descriptor_set_opts,
    );

    unsafe {
        let mut layout_create_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

        let push_constant_ranges = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::ALL_GRAPHICS,
            offset: 0,
            size: desc.push_constants_bytes as _,
        };

        if desc.push_constants_bytes > 0 {
            layout_create_info = layout_create_info
                .push_constant_ranges(std::slice::from_ref(&push_constant_ranges));
        }

        let pipeline_layout = device
            .raw
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let entry_names = TempList::new();
        let shader_stage_create_infos: Vec<_> = shaders
            .iter()
            .map(|desc| {
                let shader_info = vk::ShaderModuleCreateInfo::builder()
                    .code(desc.code.as_slice_of::<u32>().unwrap());

                let shader_module = device
                    .raw
                    .create_shader_module(&shader_info, None)
                    .expect("Shader module error");

                let stage = match desc.desc.stage {
                    ShaderPipelineStage::Vertex => vk::ShaderStageFlags::VERTEX,
                    ShaderPipelineStage::Pixel => vk::ShaderStageFlags::FRAGMENT,
                    _ => unimplemented!(),
                };

                vk::PipelineShaderStageCreateInfo::builder()
                    .module(shader_module)
                    .name(entry_names.add(CString::new(desc.desc.entry.as_str()).unwrap()))
                    .stage(stage)
                    .build()
            })
            .collect();

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: std::ptr::null(),
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: std::ptr::null(),
            ..Default::default()
        };
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: if desc.face_cull {
                ash::vk::CullModeFlags::BACK
            } else {
                ash::vk::CullModeFlags::NONE
            },
            ..Default::default()
        };
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: 1,
            depth_write_enable: if desc.depth_write { 1 } else { 0 },
            depth_compare_op: vk::CompareOp::GREATER_OR_EQUAL,
            front: noop_stencil_state,
            back: noop_stencil_state,
            max_depth_bounds: 1.0,
            ..Default::default()
        };

        let color_attachment_count = desc.render_pass.framebuffer_cache.color_attachment_count;

        let color_blend_attachment_states = vec![
            vk::PipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ZERO,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::all(),
            };
            color_attachment_count
        ];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(desc.render_pass.raw);

        let pipeline = device
            .raw
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphic_pipeline_info.build()],
                None,
            )
            .expect("Unable to create graphics pipeline")[0];

        let mut descriptor_pool_sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for bindings in set_layout_info.iter() {
            for ty in bindings.values() {
                if let Some(mut dps) = descriptor_pool_sizes.iter_mut().find(|item| item.ty == *ty)
                {
                    dps.descriptor_count += 1;
                } else {
                    descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                        ty: *ty,
                        descriptor_count: 1,
                    })
                }
            }
        }

        Ok(RasterPipeline {
            common: ShaderPipelineCommon {
                pipeline_layout,
                pipeline,
                //render_pass: desc.render_pass.clone(),
                set_layout_info,
                descriptor_pool_sizes,
                descriptor_set_layouts,
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            },
        })
    }
}

fn merge_shader_stage_layout_pair(
    src: StageDescriptorSetLayouts,
    dst: &mut StageDescriptorSetLayouts,
) {
    for (set_idx, set) in src.into_iter() {
        match dst.entry(set_idx) {
            Entry::Occupied(mut existing) => {
                let existing = existing.get_mut();
                for (binding_idx, binding) in set {
                    match existing.entry(binding_idx) {
                        Entry::Occupied(existing) => {
                            let existing = existing.get();
                            assert_eq!(
                                existing.ty, binding.ty,
                                "binding idx: {}, name: {:?}",
                                binding_idx, binding.name
                            );
                            assert_eq!(
                                existing.name, binding.name,
                                "binding idx: {}, name: {:?}",
                                binding_idx, binding.name
                            );
                        }
                        Entry::Vacant(vacant) => {
                            vacant.insert(binding);
                        }
                    }
                }
            }
            Entry::Vacant(vacant) => {
                vacant.insert(set);
            }
        }
    }
}

pub(crate) fn merge_shader_stage_layouts(
    stages: Vec<StageDescriptorSetLayouts>,
) -> StageDescriptorSetLayouts {
    let mut stages = stages.into_iter();
    let mut result = stages.next().unwrap_or_default();

    for stage in stages {
        merge_shader_stage_layout_pair(stage, &mut result);
    }

    result
}
