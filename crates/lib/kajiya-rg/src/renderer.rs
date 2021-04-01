use crate::{
    CompiledRenderGraph, ExportedHandle, ExportedTemporalRenderGraphState, PredefinedDescriptorSet,
    RenderGraphExecutionParams, TemporalRenderGraph, TemporalRenderGraphState,
    TemporalResourceState,
};
use kajiya_backend::{
    ash::{version::DeviceV1_0, vk},
    dynamic_constants::*,
    gpu_allocator::MemoryLocation,
    pipeline_cache::*,
    rspirv_reflect,
    transient_resource_cache::TransientResourceCache,
    vk_sync,
    vulkan::{
        self, image::*, presentation::blit_image_to_swapchain, shader::*, swapchain::Swapchain,
        RenderBackend,
    },
    Device,
};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::{collections::HashMap, sync::Arc};
use turbosloth::*;
use vulkan::buffer::{Buffer, BufferDesc};

enum TemporalRg {
    Inert(TemporalRenderGraphState),
    Exported(ExportedTemporalRenderGraphState),
}

impl Default for TemporalRg {
    fn default() -> Self {
        Self::Inert(Default::default())
    }
}

pub struct Renderer {
    device: Arc<Device>,

    pipeline_cache: PipelineCache,
    transient_resource_cache: TransientResourceCache,
    dynamic_constants: DynamicConstants,
    frame_descriptor_set: vk::DescriptorSet,

    present_shader: ComputePipeline,

    compiled_rg: Option<CompiledRenderGraph>,
    rg_output: Option<RenderGraphOutput>,
    temporal_rg_state: TemporalRg,
}

lazy_static::lazy_static! {
    static ref FRAME_CONSTANTS_LAYOUT: HashMap<u32, rspirv_reflect::DescriptorInfo> = [(
        0,
        rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::UNIFORM_BUFFER,
            is_bindless: false,
            /*stages: rspirv_reflect::ShaderStageFlags(
                (vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::ALL_GRAPHICS
                    | vk::ShaderStageFlags::ALL)
                    .as_raw(),
            ),*/
            name: Default::default(),
        },
    )]
    .iter()
    .cloned()
    .collect();
}

pub struct RenderGraphOutput {
    pub main_img: ExportedHandle<Image>,
    pub ui_img: ExportedHandle<Image>,
}

impl Renderer {
    pub fn new(backend: &RenderBackend) -> anyhow::Result<Self> {
        let present_shader = vulkan::presentation::create_present_compute_shader(&*backend.device);

        let dynamic_constants = DynamicConstants::new({
            backend
                .device
                .create_buffer_impl(
                    BufferDesc {
                        size: DYNAMIC_CONSTANTS_SIZE_BYTES * 2,
                        usage: vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        mapped: true,
                    },
                    Default::default(),
                    MemoryLocation::CpuToGpu,
                )
                .expect("a buffer for dynamic constants")
        });

        let frame_descriptor_set =
            Self::create_frame_descriptor_set(&backend, &dynamic_constants.buffer);

        Ok(Renderer {
            device: backend.device.clone(),
            dynamic_constants,
            frame_descriptor_set,
            pipeline_cache: PipelineCache::new(&LazyCache::create()),
            transient_resource_cache: Default::default(),
            present_shader,

            compiled_rg: None,
            rg_output: None,

            temporal_rg_state: Default::default(),
        })
    }

    pub fn draw_frame<PrepareFrameConstantsFn>(
        &mut self,
        //render_client: &mut dyn RenderClient<FrameState>,
        prepare_frame_constants: PrepareFrameConstantsFn,
        swapchain: &mut Swapchain,
    ) where
        PrepareFrameConstantsFn: FnOnce(&mut DynamicConstants),
    {
        let frame_constants_offset = self.dynamic_constants.current_offset();
        //render_client.prepare_frame_constants(&mut self.dynamic_constants, frame_state);
        prepare_frame_constants(&mut self.dynamic_constants);

        let swapchain_extent = swapchain.extent();

        // Note: this can be done at the end of the frame, not at the start.
        // The image can be acquired just in time for a blit into it,
        // after all the other rendering commands have been recorded.
        let swapchain_image = swapchain
            .acquire_next_image()
            .ok()
            .expect("swapchain image");

        let current_frame = self.device.current_frame();
        let cb = &current_frame.command_buffer;
        let device = &*self.device;
        let raw_device = &device.raw;

        unsafe {
            raw_device
                .begin_command_buffer(
                    cb.raw,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            current_frame.profiler_data.begin_frame(device, cb.raw);

            if let Some((rg, rg_output)) = self.compiled_rg.take().zip(self.rg_output.take()) {
                let retired_rg = rg.execute(
                    RenderGraphExecutionParams {
                        device: &self.device,
                        pipeline_cache: &mut self.pipeline_cache,
                        frame_descriptor_set: self.frame_descriptor_set,
                        frame_constants_offset,
                        profiler_data: &current_frame.profiler_data,
                    },
                    &mut self.transient_resource_cache,
                    &mut self.dynamic_constants,
                    cb,
                );

                let (main_img, main_img_access_type) =
                    retired_rg.exported_resource(rg_output.main_img);
                assert!(
                    main_img_access_type
                        == vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer
                );

                let (ui_img, ui_img_access_type) = retired_rg.exported_resource(rg_output.ui_img);
                assert!(
                    ui_img_access_type
                        == vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer
                );

                self.temporal_rg_state = match std::mem::take(&mut self.temporal_rg_state) {
                    TemporalRg::Inert(_) => {
                        panic!("Trying to retire the render graph, but it's inert. Was prepare_frame not caled?");
                    }
                    TemporalRg::Exported(rg) => TemporalRg::Inert(rg.retire_temporal(&retired_rg)),
                };

                blit_image_to_swapchain(
                    main_img.desc.extent_2d(),
                    swapchain_extent,
                    &*self.device,
                    cb,
                    &swapchain_image,
                    main_img.view(device, &ImageViewDesc::default()),
                    ui_img.view(device, &ImageViewDesc::default()),
                    &self.present_shader,
                );

                retired_rg.release_resources(&mut self.transient_resource_cache);
            }

            current_frame.profiler_data.finish_frame(device, cb.raw);
            raw_device.end_command_buffer(cb.raw).unwrap();
        }

        self.dynamic_constants.advance_frame();

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(std::slice::from_ref(&swapchain_image.acquire_semaphore))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COMPUTE_SHADER])
            .command_buffers(std::slice::from_ref(&cb.raw));

        unsafe {
            raw_device
                .reset_fences(std::slice::from_ref(&cb.submit_done_fence))
                .expect("reset_fences");

            //println!("queue_submit");
            raw_device
                .queue_submit(
                    self.device.universal_queue.raw,
                    &[submit_info.build()],
                    cb.submit_done_fence,
                )
                .expect("queue submit failed.");
        }

        swapchain.present_image(swapchain_image, &[]);
        self.device.finish_frame(current_frame);
    }

    // Descriptor set for per-frame data
    fn create_frame_descriptor_set(
        backend: &RenderBackend,
        dynamic_constants: &Buffer,
    ) -> vk::DescriptorSet {
        let device = &backend.device.raw;

        let set_binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND];

        let mut binding_flags_create_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&set_binding_flags)
                .build();

        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&[vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .stage_flags(vk::ShaderStageFlags::ALL)
                            .binding(0)
                            .build()])
                        .push_next(&mut binding_flags_create_info)
                        .build(),
                    None,
                )
                .unwrap()
        };

        let descriptor_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 1,
        }];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&descriptor_sizes)
            .max_sets(1);

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        };

        let set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                        .build(),
                )
                .unwrap()[0]
        };

        {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(dynamic_constants.raw)
                .range(16384)
                .build();

            let write_descriptor_set = vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(std::slice::from_ref(&buffer_info))
                .build();

            unsafe {
                device.update_descriptor_sets(std::slice::from_ref(&write_descriptor_set), &[])
            };
        }

        set
    }

    pub fn prepare_frame<PrepareRenderGraphFn>(
        &mut self,
        prepare_render_graph: PrepareRenderGraphFn,
    ) -> anyhow::Result<()>
    where
        PrepareRenderGraphFn: FnOnce(&mut TemporalRenderGraph) -> RenderGraphOutput,
    {
        let mut rg = TemporalRenderGraph::new(
            match &self.temporal_rg_state {
                TemporalRg::Inert(state) => state.clone_assuming_inert(),
                TemporalRg::Exported(_) => {
                    panic!("Trying to prepare_frame but render graph is still active")
                }
            },
            self.device.clone(),
        );

        rg.predefined_descriptor_set_layouts.insert(
            2,
            PredefinedDescriptorSet {
                bindings: FRAME_CONSTANTS_LAYOUT.clone(),
            },
        );

        self.rg_output = Some(prepare_render_graph(&mut rg));

        let (rg, temporal_rg_state) = rg.export_temporal();

        self.compiled_rg = Some(rg.compile(&mut self.pipeline_cache));

        match self.pipeline_cache.prepare_frame(&self.device) {
            Ok(()) => {
                // If the frame preparation succeded, update stored temporal rg state and finish
                self.temporal_rg_state = TemporalRg::Exported(temporal_rg_state);
                Ok(())
            }
            Err(err) => {
                // If frame preparation failed, we're not going to render anything, but we've potentially created
                // some temporal resources, and we can reuse them in the next attempt.
                //
                // Import any new resources into our temporal rg state, but reset their access modes.

                let self_temporal_rg_state = match &mut self.temporal_rg_state {
                    TemporalRg::Inert(state) => state,
                    TemporalRg::Exported(_) => unreachable!(),
                };

                for (res_key, res) in temporal_rg_state.0.resources {
                    // `insert` is infrequent here, and we can avoid cloning the key.
                    #[allow(clippy::map_entry)]
                    if !self_temporal_rg_state.resources.contains_key(&res_key) {
                        let res = match res {
                            res @ TemporalResourceState::Inert { .. } => res,
                            TemporalResourceState::Imported { resource, .. }
                            | TemporalResourceState::Exported { resource, .. } => {
                                TemporalResourceState::Inert {
                                    resource,
                                    access_type: vk_sync::AccessType::Nothing,
                                }
                            }
                        };

                        self_temporal_rg_state.resources.insert(res_key, res);
                    }
                }

                Err(err)
            }
        }
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
