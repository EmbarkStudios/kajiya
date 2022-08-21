use std::{cell::UnsafeCell, sync::Arc};

use arrayvec::ArrayVec;

use super::{
    Buffer, GpuRt, GpuSrv, GpuUav, GraphRawResourceHandle, Image, Ref, ResourceRegistry,
    RgComputePipelineHandle, RgRasterPipelineHandle, RgRtPipelineHandle,
};

use kajiya_backend::{
    ash::vk,
    chunky_list::TempList,
    dynamic_constants::{
        DynamicConstants, MAX_DYNAMIC_CONSTANTS_BYTES_PER_DISPATCH,
        MAX_DYNAMIC_CONSTANTS_STORAGE_BUFFER_BYTES,
    },
    vulkan::{
        device::{CommandBuffer, Device},
        image::*,
        ray_tracing::{RayTracingAcceleration, RayTracingPipeline},
        shader::{
            ComputePipeline, FramebufferCacheKey, RasterPipeline, ShaderPipelineCommon,
            MAX_COLOR_ATTACHMENTS,
        },
    },
    BackendError,
};

pub struct RenderPassApi<'a, 'exec_params, 'constants> {
    pub cb: &'a CommandBuffer,
    pub resources: &'a mut ResourceRegistry<'exec_params, 'constants>,
}

pub enum DescriptorSetBinding {
    Image(vk::DescriptorImageInfo),
    ImageArray(Vec<vk::DescriptorImageInfo>),
    Buffer(vk::DescriptorBufferInfo),
    RayTracingAcceleration(vk::AccelerationStructureKHR),
    DynamicBuffer {
        buffer: vk::DescriptorBufferInfo,
        offset: u32,
    },
    DynamicStorageBuffer {
        buffer: vk::DescriptorBufferInfo,
        offset: u32,
    },
}

#[derive(Default)]
pub struct RenderPassCommonShaderPipelineBinding<'a> {
    // TODO: fixed size
    bindings: Vec<(u32, &'a [RenderPassBinding])>,
    raw_bindings: Vec<(u32, vk::DescriptorSet)>,
}

pub struct RenderPassPipelineBinding<'a, HandleType> {
    pipeline: HandleType,
    binding: RenderPassCommonShaderPipelineBinding<'a>,
}

impl<'a, HandleType> RenderPassPipelineBinding<'a, HandleType> {
    pub fn new(pipeline: HandleType) -> Self {
        Self {
            pipeline,
            binding: Default::default(),
        }
    }

    pub fn descriptor_set(mut self, set_idx: u32, bindings: &'a [RenderPassBinding]) -> Self {
        self.binding.bindings.push((set_idx, bindings));
        self
    }

    pub fn raw_descriptor_set(mut self, set_idx: u32, binding: vk::DescriptorSet) -> Self {
        self.binding.raw_bindings.push((set_idx, binding));
        self
    }
}

pub trait IntoRenderPassPipelineBinding: Sized {
    fn into_binding<'a>(self) -> RenderPassPipelineBinding<'a, Self>;
}

impl IntoRenderPassPipelineBinding for RgComputePipelineHandle {
    fn into_binding<'a>(self) -> RenderPassPipelineBinding<'a, Self> {
        RenderPassPipelineBinding::new(self)
    }
}

impl IntoRenderPassPipelineBinding for RgRasterPipelineHandle {
    fn into_binding<'a>(self) -> RenderPassPipelineBinding<'a, Self> {
        RenderPassPipelineBinding::new(self)
    }
}

impl IntoRenderPassPipelineBinding for RgRtPipelineHandle {
    fn into_binding<'a>(self) -> RenderPassPipelineBinding<'a, Self> {
        RenderPassPipelineBinding::new(self)
    }
}

impl<'a, 'exec_params, 'constants> RenderPassApi<'a, 'exec_params, 'constants> {
    pub fn device(&self) -> &Device {
        self.resources.execution_params.device
    }

    pub fn dynamic_constants(&mut self) -> &mut DynamicConstants {
        self.resources.dynamic_constants
    }

    pub fn bind_compute_pipeline<'s>(
        &'s mut self,
        binding: RenderPassPipelineBinding<'_, RgComputePipelineHandle>,
    ) -> Result<BoundComputePipeline<'s, 'a, 'exec_params, 'constants>, BackendError> {
        let device = self.resources.execution_params.device;
        let pipeline_arc = self.resources.compute_pipeline(binding.pipeline);

        self.bind_pipeline_common(device, pipeline_arc.as_ref(), &binding.binding)?;

        Ok(BoundComputePipeline {
            api: self,
            pipeline: pipeline_arc,
        })
    }

    pub fn bind_raster_pipeline<'s>(
        &'s self,
        binding: RenderPassPipelineBinding<'_, RgRasterPipelineHandle>,
    ) -> Result<BoundRasterPipeline<'s, 'a, 'exec_params, 'constants>, BackendError> {
        let device = self.resources.execution_params.device;
        let pipeline_arc = self.resources.raster_pipeline(binding.pipeline);

        self.bind_pipeline_common(device, pipeline_arc.as_ref(), &binding.binding)?;

        Ok(BoundRasterPipeline {
            api: self,
            pipeline: pipeline_arc,
        })
    }

    pub fn bind_ray_tracing_pipeline<'s>(
        &'s mut self,
        binding: RenderPassPipelineBinding<'_, RgRtPipelineHandle>,
    ) -> Result<BoundRayTracingPipeline<'s, 'a, 'exec_params, 'constants>, BackendError> {
        let device = self.resources.execution_params.device;
        let pipeline_arc = self.resources.ray_tracing_pipeline(binding.pipeline);

        self.bind_pipeline_common(device, pipeline_arc.as_ref(), &binding.binding)?;

        Ok(BoundRayTracingPipeline {
            api: self,
            pipeline: pipeline_arc,
        })
    }

    fn bind_pipeline_common(
        &self,
        device: &Device,
        pipeline: &ShaderPipelineCommon,
        binding: &RenderPassCommonShaderPipelineBinding,
    ) -> Result<(), BackendError> {
        unsafe {
            device.raw.cmd_bind_pipeline(
                self.cb.raw,
                pipeline.pipeline_bind_point,
                pipeline.pipeline,
            );
        }

        // Bind frame constants
        if pipeline
            .set_layout_info
            .get(2)
            .map(|set| !set.is_empty())
            .unwrap_or_default()
        {
            unsafe {
                device.raw.cmd_bind_descriptor_sets(
                    self.cb.raw,
                    pipeline.pipeline_bind_point,
                    pipeline.pipeline_layout,
                    2,
                    &[self.resources.execution_params.frame_descriptor_set],
                    &[
                        self.resources
                            .execution_params
                            .frame_constants_layout
                            .globals_offset,
                        self.resources
                            .execution_params
                            .frame_constants_layout
                            .instance_dynamic_parameters_offset,
                        self.resources
                            .execution_params
                            .frame_constants_layout
                            .triangle_lights_offset,
                    ],
                );
            }
        }

        for (set_idx, bindings) in &binding.bindings {
            let set_idx = *set_idx;
            if pipeline.set_layout_info.get(set_idx as usize).is_none() {
                continue;
            }

            let bindings: Result<Vec<_>, BackendError> = bindings
                .iter()
                .map(|binding| {
                    Ok(match binding {
                        RenderPassBinding::Image(image) => DescriptorSetBinding::Image(
                            vk::DescriptorImageInfo::builder()
                                .image_layout(image.image_layout)
                                .image_view(
                                    self.resources.image_view(image.handle, &image.view_desc)?,
                                )
                                .build(),
                        ),
                        RenderPassBinding::ImageArray(images) => DescriptorSetBinding::ImageArray(
                            images
                                .iter()
                                .map(|image| {
                                    Ok(vk::DescriptorImageInfo::builder()
                                        .image_layout(image.image_layout)
                                        .image_view(
                                            self.resources
                                                .image_view(image.handle, &image.view_desc)?,
                                        )
                                        .build())
                                })
                                .collect::<Result<Vec<_>, BackendError>>()?,
                        ),
                        RenderPassBinding::Buffer(buffer) => DescriptorSetBinding::Buffer(
                            vk::DescriptorBufferInfo::builder()
                                .buffer(
                                    self.resources
                                        .buffer_from_raw_handle::<GpuSrv>(buffer.handle)
                                        .raw,
                                )
                                .range(vk::WHOLE_SIZE)
                                .build(),
                        ),
                        RenderPassBinding::RayTracingAcceleration(acc) => {
                            DescriptorSetBinding::RayTracingAcceleration(
                                self.resources
                                    .rt_acceleration_from_raw_handle::<GpuSrv>(acc.handle)
                                    .raw,
                            )
                        }
                        RenderPassBinding::DynamicConstants(offset) => {
                            DescriptorSetBinding::DynamicBuffer {
                                buffer: vk::DescriptorBufferInfo::builder()
                                    .buffer(self.resources.dynamic_constants.buffer.raw)
                                    .range(MAX_DYNAMIC_CONSTANTS_BYTES_PER_DISPATCH as u64)
                                    .build(),
                                offset: *offset,
                            }
                        }
                        RenderPassBinding::DynamicConstantsStorageBuffer(offset) => {
                            DescriptorSetBinding::DynamicStorageBuffer {
                                buffer: vk::DescriptorBufferInfo::builder()
                                    .buffer(self.resources.dynamic_constants.buffer.raw)
                                    .range(MAX_DYNAMIC_CONSTANTS_STORAGE_BUFFER_BYTES as u64)
                                    .build(),
                                offset: *offset,
                            }
                        }
                    })
                })
                .collect();
            let bindings = bindings?;

            bind_descriptor_set(
                self.resources.execution_params.device,
                self.cb,
                &pipeline,
                set_idx,
                &bindings,
            );
        }

        for (set_idx, binding) in &binding.raw_bindings {
            let set_idx = *set_idx;
            if pipeline.set_layout_info.get(set_idx as usize).is_none() {
                continue;
            }

            unsafe {
                self.resources
                    .execution_params
                    .device
                    .raw
                    .cmd_bind_descriptor_sets(
                        self.cb.raw,
                        pipeline.pipeline_bind_point,
                        pipeline.pipeline_layout,
                        set_idx,
                        std::slice::from_ref(binding),
                        &[],
                    );
            }
        }

        Ok(())
    }

    pub fn begin_render_pass(
        &mut self,
        render_pass: &kajiya_backend::vulkan::shader::RenderPass,
        dims: [u32; 2],
        color_attachments: &[(Ref<Image, GpuRt>, &ImageViewDesc)],
        depth_attachment: Option<(Ref<Image, GpuRt>, &ImageViewDesc)>,
    ) -> Result<(), BackendError> {
        let device = self.resources.execution_params.device;

        let framebuffer = render_pass
            .framebuffer_cache
            .get_or_create(
                &device.raw,
                FramebufferCacheKey::new(
                    dims,
                    color_attachments.iter().map(|(a, _)| {
                        &self.resources.image_from_raw_handle::<GpuRt>(a.handle).desc
                    }),
                    depth_attachment.as_ref().map(|(a, _)| {
                        &self.resources.image_from_raw_handle::<GpuRt>(a.handle).desc
                    }),
                ),
            )
            .unwrap();

        // Bind images to the imageless framebuffer
        let image_attachments: Result<
            ArrayVec<[vk::ImageView; MAX_COLOR_ATTACHMENTS + 1]>,
            BackendError,
        > = color_attachments
            .iter()
            .chain(depth_attachment.as_ref().into_iter())
            .map(|(img, view)| self.resources.image_view(img.handle, view))
            .collect();
        let image_attachments = image_attachments?;

        let mut pass_attachment_desc =
            vk::RenderPassAttachmentBeginInfoKHR::builder().attachments(&image_attachments);

        let [width, height] = dims;

        //.clear_values(&clear_values)
        let pass_begin_desc = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.raw)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: width as _,
                    height: height as _,
                },
            })
            .push_next(&mut pass_attachment_desc);

        unsafe {
            device.raw.cmd_begin_render_pass(
                self.cb.raw,
                &pass_begin_desc,
                vk::SubpassContents::INLINE,
            );
        }

        Ok(())
    }

    pub fn end_render_pass(&mut self) {
        let device = self.resources.execution_params.device;
        unsafe {
            device.raw.cmd_end_render_pass(self.cb.raw);
        }
    }

    pub fn set_default_view_and_scissor(&mut self, [width, height]: [u32; 2]) {
        let raw_device = &self.resources.execution_params.device.raw;
        let cb_raw = self.cb.raw;

        unsafe {
            raw_device.cmd_set_viewport(
                cb_raw,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: (height as f32),
                    width: width as _,
                    height: -(height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );

            raw_device.cmd_set_scissor(
                cb_raw,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: width as _,
                        height: height as _,
                    },
                }],
            );
        }
    }
}

pub struct BoundComputePipeline<'api, 'a, 'exec_params, 'constants> {
    api: &'api mut RenderPassApi<'a, 'exec_params, 'constants>,
    pipeline: Arc<ComputePipeline>,
}

impl<'api, 'a, 'exec_params, 'constants> BoundComputePipeline<'api, 'a, 'exec_params, 'constants> {
    pub fn dispatch(&self, threads: [u32; 3]) {
        let group_size = self.pipeline.group_size;

        unsafe {
            self.api.device().raw.cmd_dispatch(
                self.api.cb.raw,
                (threads[0] + group_size[0] - 1) / group_size[0],
                (threads[1] + group_size[1] - 1) / group_size[1],
                (threads[2] + group_size[2] - 1) / group_size[2],
            );
        }
    }

    pub fn dispatch_indirect(&self, args_buffer: Ref<Buffer, GpuSrv>, args_buffer_offset: u64) {
        unsafe {
            self.api.device().raw.cmd_dispatch_indirect(
                self.api.cb.raw,
                self.api.resources.buffer(args_buffer).raw,
                args_buffer_offset,
            );
        }
    }

    pub fn push_constants(&self, command_buffer: vk::CommandBuffer, offset: u32, constants: &[u8]) {
        unsafe {
            self.api
                .resources
                .execution_params
                .device
                .raw
                .cmd_push_constants(
                    command_buffer,
                    self.pipeline.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    offset,
                    constants,
                )
        }
    }
}

pub struct BoundRasterPipeline<'api, 'a, 'exec_params, 'constants> {
    api: &'api RenderPassApi<'a, 'exec_params, 'constants>,
    pipeline: Arc<RasterPipeline>,
}

impl<'api, 'a, 'exec_params, 'constants> BoundRasterPipeline<'api, 'a, 'exec_params, 'constants> {
    pub fn push_constants(
        &self,
        command_buffer: vk::CommandBuffer,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        unsafe {
            self.api
                .resources
                .execution_params
                .device
                .raw
                .cmd_push_constants(
                    command_buffer,
                    self.pipeline.pipeline_layout,
                    stage_flags,
                    offset,
                    constants,
                )
        }
    }
}

pub struct RenderPassImageBinding {
    handle: GraphRawResourceHandle,
    view_desc: ImageViewDesc,
    image_layout: vk::ImageLayout,
}

pub struct RenderPassBufferBinding {
    handle: GraphRawResourceHandle,
}

pub struct RenderPassRayTracingAccelerationBinding {
    handle: GraphRawResourceHandle,
}

pub enum RenderPassBinding {
    Image(RenderPassImageBinding),
    ImageArray(Vec<RenderPassImageBinding>),
    Buffer(RenderPassBufferBinding),
    RayTracingAcceleration(RenderPassRayTracingAccelerationBinding),
    DynamicConstants(u32),
    DynamicConstantsStorageBuffer(u32),
}

pub struct BoundRayTracingPipeline<'api, 'a, 'exec_params, 'constants> {
    api: &'api mut RenderPassApi<'a, 'exec_params, 'constants>,
    pipeline: Arc<RayTracingPipeline>,
}

impl<'api, 'a, 'exec_params, 'constants>
    BoundRayTracingPipeline<'api, 'a, 'exec_params, 'constants>
{
    pub fn trace_rays(&self, threads: [u32; 3]) {
        unsafe {
            self.api.device().ray_tracing_pipeline_ext.cmd_trace_rays(
                self.api.cb.raw,
                &self.pipeline.sbt.raygen_shader_binding_table,
                &self.pipeline.sbt.miss_shader_binding_table,
                &self.pipeline.sbt.hit_shader_binding_table,
                &self.pipeline.sbt.callable_shader_binding_table,
                threads[0],
                threads[1],
                threads[2],
            );
        }
    }

    pub fn trace_rays_indirect(&self, args_buffer: Ref<Buffer, GpuSrv>, args_buffer_offset: u64) {
        let indirect_device_address = self
            .api
            .resources
            .buffer(args_buffer)
            .device_address(self.api.device())
            + args_buffer_offset;

        unsafe {
            self.api
                .device()
                .ray_tracing_pipeline_ext
                .cmd_trace_rays_indirect(
                    self.api.cb.raw,
                    std::slice::from_ref(&self.pipeline.sbt.raygen_shader_binding_table),
                    std::slice::from_ref(&self.pipeline.sbt.miss_shader_binding_table),
                    std::slice::from_ref(&self.pipeline.sbt.hit_shader_binding_table),
                    std::slice::from_ref(&self.pipeline.sbt.callable_shader_binding_table),
                    indirect_device_address,
                );
        }
    }
}

pub trait BindRgRef {
    fn bind(&self) -> RenderPassBinding;
}

impl BindRgRef for Ref<Image, GpuSrv> {
    fn bind(&self) -> RenderPassBinding {
        self.bind_view(ImageViewDescBuilder::default())
    }
}

impl Ref<Image, GpuSrv> {
    pub fn bind_view(&self, view_desc: ImageViewDescBuilder) -> RenderPassBinding {
        RenderPassBinding::Image(RenderPassImageBinding {
            handle: self.handle,
            view_desc: view_desc.build().unwrap(),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        })
    }
}

impl BindRgRef for Vec<Ref<Image, GpuSrv>> {
    fn bind(&self) -> RenderPassBinding {
        let view_desc = ImageViewDesc::default();

        RenderPassBinding::ImageArray(
            self.iter()
                .copied()
                .map(|handle| RenderPassImageBinding {
                    handle: handle.handle,
                    view_desc,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                })
                .collect(),
        )
    }
}

impl BindRgRef for Ref<Image, GpuUav> {
    fn bind(&self) -> RenderPassBinding {
        self.bind_view(ImageViewDescBuilder::default())
    }
}

impl Ref<Image, GpuUav> {
    pub fn bind_view(&self, view_desc: ImageViewDescBuilder) -> RenderPassBinding {
        RenderPassBinding::Image(RenderPassImageBinding {
            handle: self.handle,
            view_desc: view_desc.build().unwrap(),
            image_layout: vk::ImageLayout::GENERAL,
        })
    }
}

impl BindRgRef for Ref<Buffer, GpuSrv> {
    fn bind(&self) -> RenderPassBinding {
        RenderPassBinding::Buffer(RenderPassBufferBinding {
            handle: self.handle,
        })
    }
}

impl BindRgRef for Ref<Buffer, GpuUav> {
    fn bind(&self) -> RenderPassBinding {
        RenderPassBinding::Buffer(RenderPassBufferBinding {
            handle: self.handle,
        })
    }
}

impl BindRgRef for Ref<RayTracingAcceleration, GpuSrv> {
    fn bind(&self) -> RenderPassBinding {
        RenderPassBinding::RayTracingAcceleration(RenderPassRayTracingAccelerationBinding {
            handle: self.handle,
        })
    }
}

fn bind_descriptor_set(
    device: &Device,
    cb: &CommandBuffer,
    pipeline: &impl std::ops::Deref<Target = ShaderPipelineCommon>,
    set_index: u32,
    bindings: &[DescriptorSetBinding],
) {
    let shader_set_info = if let Some(info) = pipeline.set_layout_info.get(set_index as usize) {
        info
    } else {
        println!(
            "bind_descriptor_set: set index {} does not exist",
            set_index
        );
        return;
    };

    let image_info = TempList::new();
    let buffer_info = TempList::new();
    let accel_info: TempList<UnsafeCell<vk::WriteDescriptorSetAccelerationStructureKHR>> =
        TempList::new();

    let raw_device = &device.raw;

    let descriptor_pool = {
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pipeline.descriptor_pool_sizes);

        unsafe { raw_device.create_descriptor_pool(&descriptor_pool_create_info, None) }.unwrap()
    };
    device.defer_release(descriptor_pool);

    let descriptor_set = {
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(
                &pipeline.descriptor_set_layouts[set_index as usize],
            ));

        unsafe { raw_device.allocate_descriptor_sets(&descriptor_set_allocate_info) }.unwrap()[0]
    };

    unsafe {
        let mut dynamic_offsets: Vec<u32> = Vec::new();
        let descriptor_writes: Vec<vk::WriteDescriptorSet> =
            bindings
                .iter()
                .enumerate()
                .filter(|(binding_idx, _)| shader_set_info.contains_key(&(*binding_idx as u32)))
                .map(|(binding_idx, binding)| {
                    let write = vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set)
                        .dst_binding(binding_idx as _)
                        .dst_array_element(0);

                    match binding {
                        DescriptorSetBinding::Image(image) => write
                            .descriptor_type(match image.image_layout {
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
                                    vk::DescriptorType::SAMPLED_IMAGE
                                }
                                vk::ImageLayout::GENERAL => vk::DescriptorType::STORAGE_IMAGE,
                                _ => unimplemented!("{:?}", image.image_layout),
                            })
                            .image_info(std::slice::from_ref(image_info.add(*image)))
                            .build(),
                        DescriptorSetBinding::ImageArray(images) => {
                            assert!(!images.is_empty());

                            write
                                .descriptor_type(match images[0].image_layout {
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
                                        vk::DescriptorType::SAMPLED_IMAGE
                                    }
                                    vk::ImageLayout::GENERAL => vk::DescriptorType::STORAGE_IMAGE,
                                    _ => unimplemented!("{:?}", images[0].image_layout),
                                })
                                .image_info(images.as_slice())
                                .build()
                        }
                        DescriptorSetBinding::Buffer(buffer) => write
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(buffer_info.add(*buffer)))
                            .build(),
                        DescriptorSetBinding::DynamicBuffer { buffer, offset } => {
                            dynamic_offsets.push(*offset);
                            write
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                                .buffer_info(std::slice::from_ref(buffer_info.add(*buffer)))
                                .build()
                        }
                        DescriptorSetBinding::DynamicStorageBuffer { buffer, offset } => {
                            dynamic_offsets.push(*offset);
                            write
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                                .buffer_info(std::slice::from_ref(buffer_info.add(*buffer)))
                                .build()
                        }
                        DescriptorSetBinding::RayTracingAcceleration(acc) => {
                            let mut write = write
                            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                            .push_next(
                                accel_info.add(UnsafeCell::new(
                                    vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                                        .acceleration_structures(std::slice::from_ref(acc))
                                        .build(),
                                )).get().as_mut().unwrap(),
                            )
                            .build();

                            // This is only set by the builder for images, buffers, or views; need to set explicitly after
                            write.descriptor_count = 1;
                            write
                        }
                    }
                })
                .collect();

        device.raw.update_descriptor_sets(&descriptor_writes, &[]);

        device.raw.cmd_bind_descriptor_sets(
            cb.raw,
            pipeline.pipeline_bind_point,
            pipeline.pipeline_layout,
            set_index,
            &[descriptor_set],
            dynamic_offsets.as_slice(),
        );
    }
}
