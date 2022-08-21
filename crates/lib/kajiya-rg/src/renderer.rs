use crate::{
    CompiledRenderGraph, ExecutingRenderGraph, ExportedTemporalRenderGraphState,
    PredefinedDescriptorSet, RenderGraphExecutionParams, TemporalRenderGraph,
    TemporalRenderGraphState, TemporalResourceState,
};
use kajiya_backend::{
    ash::vk,
    dynamic_constants::*,
    pipeline_cache::*,
    rspirv_reflect,
    transient_resource_cache::TransientResourceCache,
    vk_sync,
    vulkan::{self, swapchain::Swapchain, RenderBackend},
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

    compiled_rg: Option<CompiledRenderGraph>,
    temporal_rg_state: TemporalRg,
}

lazy_static::lazy_static! {
    static ref FRAME_CONSTANTS_LAYOUT: HashMap<u32, rspirv_reflect::DescriptorInfo> = [
    // frame_constants
    (
        0,
        rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::UNIFORM_BUFFER,
            dimensionality: rspirv_reflect::DescriptorDimensionality::Single,
            name: Default::default(),
        },
    ),
    // instance_dynamic_parameters_dyn
    (
        1,
        rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            dimensionality: rspirv_reflect::DescriptorDimensionality::Single,
            name: Default::default(),
        },
    ),
    // triangle_lights_dyn
    (
        2,
        rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            dimensionality: rspirv_reflect::DescriptorDimensionality::Single,
            name: Default::default(),
        },
    ),
    ]
    .iter()
    .cloned()
    .collect();
}

pub struct FrameConstantsLayout {
    pub globals_offset: u32,
    pub instance_dynamic_parameters_offset: u32,
    pub triangle_lights_offset: u32,
}

impl Renderer {
    pub fn new(backend: &RenderBackend) -> anyhow::Result<Self> {
        let dynamic_constants = DynamicConstants::new({
            backend.device.create_buffer(
                BufferDesc::new_cpu_to_gpu(
                    DYNAMIC_CONSTANTS_SIZE_BYTES * DYNAMIC_CONSTANTS_BUFFER_COUNT,
                    vk::BufferUsageFlags::UNIFORM_BUFFER
                        | vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                ),
                "dynamic constants buffer",
                None,
            )?
        });

        let frame_descriptor_set =
            Self::create_frame_descriptor_set(backend, &dynamic_constants.buffer);

        Ok(Renderer {
            device: backend.device.clone(),
            dynamic_constants,
            frame_descriptor_set,
            pipeline_cache: PipelineCache::new(&LazyCache::create()),
            transient_resource_cache: Default::default(),

            compiled_rg: None,
            temporal_rg_state: Default::default(),
        })
    }

    pub fn draw_frame<PrepareFrameConstantsFn>(
        &mut self,
        prepare_frame_constants: PrepareFrameConstantsFn,
        swapchain: &mut Swapchain,
    ) where
        PrepareFrameConstantsFn: FnOnce(&mut DynamicConstants) -> FrameConstantsLayout,
    {
        let rg = if let Some(rg) = self.compiled_rg.take() {
            rg
        } else {
            return;
        };

        let device = &*self.device;
        let raw_device = &device.raw;

        let current_frame = self.device.begin_frame();

        // Both command buffers are accessible now, so begin recording.
        for cb in [
            &current_frame.main_command_buffer,
            &current_frame.presentation_command_buffer,
        ] {
            unsafe {
                raw_device
                    .reset_command_buffer(cb.raw, vk::CommandBufferResetFlags::default())
                    .unwrap();

                raw_device
                    .begin_command_buffer(
                        cb.raw,
                        &vk::CommandBufferBeginInfo::builder()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap();
            }
        }

        // Now that we can write to GPU data, prepare global frame constants.
        let frame_constants_layout = prepare_frame_constants(&mut self.dynamic_constants);

        let mut executing_rg: ExecutingRenderGraph;

        // Record and submit the main command buffer
        {
            let main_cb = &current_frame.main_command_buffer;

            current_frame
                .profiler_data
                .begin_frame(&device.raw, main_cb.raw);

            executing_rg = {
                puffin::profile_scope!("rg begin_execute");

                rg.begin_execute(
                    RenderGraphExecutionParams {
                        device: &self.device,
                        pipeline_cache: &mut self.pipeline_cache,
                        frame_descriptor_set: self.frame_descriptor_set,
                        frame_constants_layout,
                        profiler_data: &current_frame.profiler_data,
                    },
                    &mut self.transient_resource_cache,
                    &mut self.dynamic_constants,
                )
            };

            // Record and submit the main command buffer
            unsafe {
                puffin::profile_scope!("main cb");

                {
                    puffin::profile_scope!("rg::record_main_cb");
                    executing_rg.record_main_cb(main_cb)
                }

                raw_device.end_command_buffer(main_cb.raw).unwrap();

                let submit_info = [vk::SubmitInfo::builder()
                    .command_buffers(std::slice::from_ref(&main_cb.raw))
                    .build()];

                raw_device
                    .reset_fences(std::slice::from_ref(&main_cb.submit_done_fence))
                    .expect("reset_fences");

                puffin::profile_scope!("submit main cb");

                // Try to submit the command buffer to the GPU. We might encounter a GPU crash.
                raw_device
                    .queue_submit(
                        self.device.universal_queue.raw,
                        &submit_info,
                        main_cb.submit_done_fence,
                    )
                    .map_err(|err| device.report_error(err.into()))
                    .expect("main queue_submit failed");
            };
        }

        // Now that we've done the main submission and the GPU is busy, acquire the presentation image.
        // This can block, so we're doing it as late as possible.

        let swapchain_image = swapchain
            .acquire_next_image()
            .ok()
            .expect("swapchain image");

        // Execute the rest of the render graph, and submit the presentation command buffer.
        let retired_rg = {
            puffin::profile_scope!("presentation cb");

            let presentation_cb = &current_frame.presentation_command_buffer;

            // Transition the swapchain to CS write
            vulkan::barrier::record_image_barrier(
                device,
                presentation_cb.raw,
                vulkan::barrier::ImageBarrier::new(
                    swapchain_image.image.raw,
                    vk_sync::AccessType::Present,
                    vk_sync::AccessType::ComputeShaderWrite,
                    vk::ImageAspectFlags::COLOR,
                )
                .with_discard(true),
            );

            let retired_rg =
                executing_rg.record_presentation_cb(presentation_cb, swapchain_image.image.clone());

            // Transition the swapchain to present
            vulkan::barrier::record_image_barrier(
                device,
                presentation_cb.raw,
                vulkan::barrier::ImageBarrier::new(
                    swapchain_image.image.raw,
                    vk_sync::AccessType::ComputeShaderWrite,
                    vk_sync::AccessType::Present,
                    vk::ImageAspectFlags::COLOR,
                ),
            );

            current_frame
                .profiler_data
                .end_frame(&device.raw, presentation_cb.raw);

            // Record and submit the presentation command buffer
            unsafe {
                raw_device.end_command_buffer(presentation_cb.raw).unwrap();

                let submit_info = [vk::SubmitInfo::builder()
                    .wait_semaphores(std::slice::from_ref(&swapchain_image.acquire_semaphore))
                    .signal_semaphores(std::slice::from_ref(
                        &swapchain_image.rendering_finished_semaphore,
                    ))
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COMPUTE_SHADER])
                    .command_buffers(std::slice::from_ref(&presentation_cb.raw))
                    .build()];
                raw_device
                    .reset_fences(std::slice::from_ref(&presentation_cb.submit_done_fence))
                    .expect("reset_fences");

                puffin::profile_scope!("submit presentation cb");
                raw_device
                    .queue_submit(
                        self.device.universal_queue.raw,
                        &submit_info,
                        presentation_cb.submit_done_fence,
                    )
                    .map_err(|err| device.report_error(err.into()))
                    .expect("presentation queue_submit failed");
            }

            swapchain.present_image(swapchain_image);

            retired_rg
        };

        self.temporal_rg_state = match std::mem::take(&mut self.temporal_rg_state) {
            TemporalRg::Inert(_) => {
                panic!("Trying to retire the render graph, but it's inert. Was prepare_frame not caled?");
            }
            TemporalRg::Exported(rg) => TemporalRg::Inert(rg.retire_temporal(&retired_rg)),
        };

        retired_rg.release_resources(&mut self.transient_resource_cache);

        self.dynamic_constants.advance_frame();
        self.device.finish_frame(current_frame);
    }

    // Descriptor set for per-frame data
    fn create_frame_descriptor_set(
        backend: &RenderBackend,
        dynamic_constants: &Buffer,
    ) -> vk::DescriptorSet {
        let device = &backend.device.raw;

        let set_binding_flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        ];

        let mut binding_flags_create_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&set_binding_flags)
                .build();

        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&[
                            // frame_constants
                            vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                                .stage_flags(vk::ShaderStageFlags::ALL)
                                .binding(0)
                                .build(),
                            // instance_dynamic_parameters
                            vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                                .stage_flags(vk::ShaderStageFlags::ALL)
                                .binding(1)
                                .build(),
                            // triangle_lights_dyn
                            vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                                .stage_flags(vk::ShaderStageFlags::ALL)
                                .binding(2)
                                .build(),
                        ])
                        .push_next(&mut binding_flags_create_info)
                        .build(),
                    None,
                )
                .unwrap()
        };

        let descriptor_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count: 2,
            },
        ];

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
            let uniform_buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(dynamic_constants.raw)
                .range(MAX_DYNAMIC_CONSTANTS_BYTES_PER_DISPATCH as u64)
                .build();
            let storage_buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(dynamic_constants.raw)
                .range(MAX_DYNAMIC_CONSTANTS_STORAGE_BUFFER_BYTES as u64)
                .build();

            let descriptor_set_writes = [
                // `frame_constants`
                vk::WriteDescriptorSet::builder()
                    .dst_binding(0)
                    .dst_set(set)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .buffer_info(std::slice::from_ref(&uniform_buffer_info))
                    .build(),
                // `instance_dynamic_parameters_dyn`
                vk::WriteDescriptorSet::builder()
                    .dst_binding(1)
                    .dst_set(set)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                    .buffer_info(std::slice::from_ref(&storage_buffer_info))
                    .build(),
                // `triangle_lights_dyn`
                vk::WriteDescriptorSet::builder()
                    .dst_binding(2)
                    .dst_set(set)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                    .buffer_info(std::slice::from_ref(&storage_buffer_info))
                    .build(),
            ];

            unsafe { device.update_descriptor_sets(&descriptor_set_writes, &[]) };
        }

        set
    }

    pub fn prepare_frame<PrepareRenderGraphFn>(
        &mut self,
        prepare_render_graph: PrepareRenderGraphFn,
    ) -> anyhow::Result<()>
    where
        PrepareRenderGraphFn: FnOnce(&mut TemporalRenderGraph),
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

        prepare_render_graph(&mut rg);
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
