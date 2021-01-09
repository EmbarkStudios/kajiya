use crate::{
    backend::{self, image::*, presentation::blit_image_to_swapchain, shader::*, RenderBackend},
    rg,
    rg::CompiledRenderGraph,
    rg::RenderGraph,
    rg::{RenderGraphExecutionParams, RetiredRenderGraph},
    transient_resource_cache::TransientResourceCache,
};
use crate::{dynamic_constants::*, pipeline_cache::*};
use ash::{version::DeviceV1_0, vk};
use backend::{
    barrier::record_image_barrier,
    barrier::ImageBarrier,
    buffer::{Buffer, BufferDesc},
};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::collections::HashMap;
use turbosloth::*;

pub struct Renderer {
    backend: RenderBackend,
    pipeline_cache: PipelineCache,
    transient_resource_cache: TransientResourceCache,
    dynamic_constants: DynamicConstants,
    frame_descriptor_set: vk::DescriptorSet,

    present_shader: ComputePipeline,

    compiled_rg: Option<CompiledRenderGraph>,
    rg_output_tex: Option<rg::ExportedHandle<Image>>,
}

lazy_static::lazy_static! {
    static ref FRAME_CONSTANTS_LAYOUT: HashMap<u32, rspirv_reflect::DescriptorInfo> = [(
        0,
        rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::UNIFORM_BUFFER,
            is_bindless: false,
            stages: rspirv_reflect::ShaderStageFlags(
                (vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::ALL_GRAPHICS
                    | vk::ShaderStageFlags::RAYGEN_KHR)
                    .as_raw(),
            ),
            name: Default::default(),
        },
    )]
    .iter()
    .cloned()
    .collect();
}

pub trait RenderClient<FrameState: 'static> {
    fn prepare_render_graph(
        &mut self,
        rg: &mut RenderGraph,
        frame_state: &FrameState,
    ) -> rg::ExportedHandle<Image>;

    fn prepare_frame_constants(
        &mut self,
        dynamic_constants: &mut DynamicConstants,
        frame_state: &FrameState,
    );

    fn retire_render_graph(&mut self, retired_rg: &RetiredRenderGraph);
}

impl Renderer {
    pub fn new(backend: RenderBackend) -> anyhow::Result<Self> {
        let present_shader = backend::presentation::create_present_compute_shader(&*backend.device);

        let dynamic_constants = DynamicConstants::new({
            let buffer_info = vk::BufferCreateInfo {
                // Allocate twice the size for even and odd frames
                size: (DYNAMIC_CONSTANTS_SIZE_BYTES * 2) as u64,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let buffer_mem_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::CpuToGpu,
                flags: vk_mem::AllocationCreateFlags::MAPPED,
                ..Default::default()
            };

            let (buffer, allocation, allocation_info) = backend
                .device
                .global_allocator
                .create_buffer(&buffer_info, &buffer_mem_info)
                .expect("vma::create_buffer");

            Buffer {
                raw: buffer,
                desc: BufferDesc {
                    size: buffer_info.size as _,
                    usage: buffer_info.usage,
                    mapped: true,
                },
                allocation,
                allocation_info,
            }
        });

        let frame_descriptor_set =
            Self::create_frame_descriptor_set(&backend, &dynamic_constants.buffer);

        Ok(Renderer {
            backend,
            dynamic_constants,
            frame_descriptor_set,
            pipeline_cache: PipelineCache::new(&LazyCache::create()),
            transient_resource_cache: Default::default(),
            present_shader,

            compiled_rg: None,
            rg_output_tex: None,
        })
    }

    pub fn draw_frame<FrameState: 'static>(
        &mut self,
        render_client: &mut dyn RenderClient<FrameState>,
        frame_state: &FrameState,
    ) {
        self.dynamic_constants.advance_frame();
        let frame_constants_offset = self.dynamic_constants.current_offset();

        render_client.prepare_frame_constants(&mut self.dynamic_constants, frame_state);

        // Note: this can be done at the end of the frame, not at the start.
        // The image can be acquired just in time for a blit into it,
        // after all the other rendering commands have been recorded.
        let swapchain_image = self
            .backend
            .swapchain
            .acquire_next_image()
            .ok()
            .expect("swapchain image");

        let current_frame = self.backend.device.current_frame();
        let cb = &current_frame.command_buffer;
        let device = &*self.backend.device;
        let raw_device = &device.raw;

        unsafe {
            raw_device
                .begin_command_buffer(
                    cb.raw,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            if let Some((rg, rg_output_img)) =
                self.compiled_rg.take().zip(self.rg_output_tex.take())
            {
                let retired_rg = rg.execute(
                    RenderGraphExecutionParams {
                        device: &self.backend.device,
                        pipeline_cache: &mut self.pipeline_cache,
                        frame_descriptor_set: self.frame_descriptor_set,
                        frame_constants_offset,
                    },
                    &mut self.transient_resource_cache,
                    &mut self.dynamic_constants,
                    cb,
                );

                let (rg_output_img, rg_output_access_type) = retired_rg.get_image(rg_output_img);

                render_client.retire_render_graph(&retired_rg);

                record_image_barrier(
                    device,
                    cb.raw,
                    ImageBarrier::new(
                        rg_output_img.raw,
                        rg_output_access_type,
                        vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                        vk::ImageAspectFlags::COLOR,
                    ),
                );

                blit_image_to_swapchain(
                    &*self.backend.device,
                    cb,
                    &swapchain_image,
                    rg_output_img.view(device, &ImageViewDesc::default()),
                    &self.present_shader,
                );

                retired_rg.release_resources(&mut self.transient_resource_cache);
            }

            raw_device.end_command_buffer(cb.raw).unwrap();
        }

        self.dynamic_constants
            .flush(&self.backend.device.global_allocator);

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(std::slice::from_ref(&swapchain_image.acquire_semaphore))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(std::slice::from_ref(&cb.raw));

        unsafe {
            raw_device
                .reset_fences(std::slice::from_ref(&cb.submit_done_fence))
                .expect("reset_fences");

            raw_device
                .queue_submit(
                    self.backend.device.universal_queue.raw,
                    &[submit_info.build()],
                    cb.submit_done_fence,
                )
                .expect("queue submit failed.");
        }

        self.backend.swapchain.present_image(swapchain_image, &[]);
        self.backend.device.finish_frame(current_frame);
    }

    // Descriptor set for per-frame data
    fn create_frame_descriptor_set(
        backend: &RenderBackend,
        dynamic_constants: &Buffer,
    ) -> vk::DescriptorSet {
        let device = &backend.device.raw;
        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&[vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .stage_flags(
                                vk::ShaderStageFlags::COMPUTE
                                    | vk::ShaderStageFlags::ALL_GRAPHICS
                                    | vk::ShaderStageFlags::RAYGEN_KHR,
                            )
                            .binding(0)
                            .build()])
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

    pub fn prepare_frame<FrameState: 'static>(
        &mut self,
        render_client: &mut dyn RenderClient<FrameState>,
        frame_state: &FrameState,
    ) -> anyhow::Result<()> {
        let mut rg = RenderGraph::new(Some(FRAME_CONSTANTS_LAYOUT.clone()));

        self.rg_output_tex = Some(render_client.prepare_render_graph(&mut rg, frame_state));

        self.compiled_rg = Some(rg.compile(&mut self.pipeline_cache));
        self.pipeline_cache.prepare_frame(&self.backend.device)?;

        Ok(())
    }
}
