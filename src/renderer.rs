use crate::backend::{
    self, image::*, presentation::blit_image_to_swapchain, shader::*, RenderBackend,
};
use crate::{
    chunky_list::TempList,
    dynamic_constants::*,
    pipeline_cache::*,
    shader_compiler::{CompileShader, CompiledShader},
    state_tracker::LocalImageStateTracker,
    viewport::ViewConstants,
    FrameState,
};
use arrayvec::ArrayVec;
use ash::{version::DeviceV1_0, vk};
use backend::{
    buffer::{Buffer, BufferDesc},
    device::CommandBuffer,
};
use glam::Vec2;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::{collections::HashMap, sync::Arc};
use turbosloth::*;
use winit::event::VirtualKeyCode;

const SDF_DIM: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone)]
struct FrameConstants {
    view_constants: ViewConstants,
    mouse: [f32; 4],
    frame_idx: u32,
}

#[allow(dead_code)]
pub struct Renderer {
    backend: RenderBackend,
    cs_cache: ComputePipelineCache,
    dynamic_constants: DynamicConstants,
    frame_descriptor_set: vk::DescriptorSet,
    frame_idx: u32,

    present_shader: ComputePipeline,
    output_img: ImageWithViews,

    gradients_shader: ComputePipelineHandle,
    raster_simple_render_pass: Arc<RenderPass>,
    raster_simple: RasterPipeline,

    brick_meta_buffer: Buffer,
    brick_inst_buffer: Buffer,
    sdf_img: ImageWithViews,
    gen_empty_sdf: ComputePipelineHandle,
    sdf_raymarch_gbuffer: ComputePipelineHandle,
    edit_sdf: ComputePipelineHandle,
    clear_bricks_meta: ComputePipelineHandle,
    find_sdf_bricks: ComputePipelineHandle,
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

pub enum DescriptorSetBinding {
    Image(vk::DescriptorImageInfo),
    Buffer(vk::DescriptorBufferInfo),
    Sampler,
}

pub mod view {
    use super::*;

    pub fn image_rw(view: &ImageView) -> DescriptorSetBinding {
        DescriptorSetBinding::Image(
            vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(view.raw)
                .build(),
        )
    }

    pub fn image(view: &ImageView) -> DescriptorSetBinding {
        DescriptorSetBinding::Image(
            vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view.raw)
                .build(),
        )
    }

    pub fn buffer_rw(buffer: &Buffer) -> DescriptorSetBinding {
        DescriptorSetBinding::Buffer(
            vk::DescriptorBufferInfo::builder()
                .buffer(buffer.raw)
                .range(vk::WHOLE_SIZE)
                .build(),
        )
    }

    pub fn sampler() -> DescriptorSetBinding {
        DescriptorSetBinding::Sampler
    }
}

impl Renderer {
    pub fn new(backend: RenderBackend, output_dims: [u32; 2]) -> anyhow::Result<Self> {
        let present_shader = backend::presentation::create_present_compute_shader(&*backend.device);

        let lazy_cache = LazyCache::create();
        let mut cs_cache = ComputePipelineCache::new(&lazy_cache);

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
                allocation,
                allocation_info,
            }
        });

        let frame_descriptor_set =
            Self::create_frame_descriptor_set(&backend, &dynamic_constants.buffer);

        let output_img = backend
            .device
            .create_image(
                ImageDesc::new_2d(output_dims)
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    //.format(vk::Format::R8G8B8A8_UNORM)
                    .usage(
                        vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    )
                    .build()
                    .unwrap(),
                None,
            )?
            .with_views(&backend.device);

        let gradients_shader =
            cs_cache.register("/assets/shaders/gradients.hlsl", |compiled_shader| {
                ComputePipelineDesc::builder()
                    .spirv(&compiled_shader.spirv)
                    .descriptor_set_opts(&[
                        (
                            0,
                            DescriptorSetLayoutOpts::builder()
                                .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR),
                        ),
                        (
                            2,
                            DescriptorSetLayoutOpts::builder()
                                .replace(FRAME_CONSTANTS_LAYOUT.clone()),
                        ),
                    ])
            });

        let vertex_shader = smol::block_on(
            CompileShader {
                path: "/assets/shaders/raster_simple_vs.hlsl".into(),
                profile: "vs".to_owned(),
            }
            .into_lazy()
            .eval(&lazy_cache),
        )?;

        let pixel_shader = smol::block_on(
            CompileShader {
                path: "/assets/shaders/raster_simple_ps.hlsl".into(),
                profile: "ps".to_owned(),
            }
            .into_lazy()
            .eval(&lazy_cache),
        )?;

        let raster_simple_render_pass = create_render_pass(
            &*backend.device,
            RenderPassDesc {
                color_attachments: &[RenderPassAttachmentDesc::new(
                    vk::Format::R16G16B16A16_SFLOAT,
                )
                .garbage_input()],
                depth_attachment: None,
            },
        )?;

        let raster_simple = create_raster_pipeline(
            &*backend.device,
            RasterPipelineDesc {
                shaders: &[
                    RasterShaderDesc::new(RasterStage::Vertex, &vertex_shader.spirv, "main")
                        .build()
                        .unwrap(),
                    RasterShaderDesc::new(RasterStage::Pixel, &pixel_shader.spirv, "main")
                        .build()
                        .unwrap(),
                ],
                render_pass: raster_simple_render_pass.clone(),
            },
        )?;

        let sdf_img = backend
            .device
            .create_image(
                ImageDesc::new_3d([SDF_DIM, SDF_DIM, SDF_DIM])
                    .format(vk::Format::R16_SFLOAT)
                    .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                    .build()
                    .unwrap(),
                None,
            )?
            .with_views(&backend.device);

        fn sdf_shader_ctor(compiled_shader: &CompiledShader) -> ComputePipelineDescBuilder {
            ComputePipelineDesc::builder()
                .spirv(&compiled_shader.spirv)
                .descriptor_set_opts(&[
                    (
                        0,
                        DescriptorSetLayoutOpts::builder()
                            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR),
                    ),
                    (
                        2,
                        DescriptorSetLayoutOpts::builder().replace(FRAME_CONSTANTS_LAYOUT.clone()),
                    ),
                ])
        }

        let brick_meta_buffer = backend.device.create_buffer(BufferDesc {
            size: 4,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        })?;

        let brick_inst_buffer = backend.device.create_buffer(BufferDesc {
            size: (SDF_DIM as usize).pow(3) * 4 * 4,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        })?;

        let gen_empty_sdf =
            cs_cache.register("/assets/shaders/sdf/gen_empty_sdf.hlsl", sdf_shader_ctor);
        let edit_sdf = cs_cache.register("/assets/shaders/sdf/edit_sdf.hlsl", sdf_shader_ctor);
        let clear_bricks_meta = cs_cache.register(
            "/assets/shaders/sdf/clear_bricks_meta.hlsl",
            sdf_shader_ctor,
        );
        let find_sdf_bricks =
            cs_cache.register("/assets/shaders/sdf/find_bricks.hlsl", sdf_shader_ctor);

        let sdf_raymarch_gbuffer = cs_cache.register(
            "/assets/shaders/sdf/sdf_raymarch_gbuffer.hlsl",
            sdf_shader_ctor,
        );

        Ok(Renderer {
            backend,
            dynamic_constants,
            frame_descriptor_set,
            frame_idx: !0,
            cs_cache,
            present_shader,
            output_img,
            gradients_shader,
            raster_simple_render_pass,
            raster_simple,

            brick_meta_buffer,
            brick_inst_buffer,
            sdf_img,
            gen_empty_sdf,
            sdf_raymarch_gbuffer,
            edit_sdf,
            clear_bricks_meta,
            find_sdf_bricks,
        })
    }

    pub fn draw_frame(&mut self, frame_state: FrameState) {
        self.dynamic_constants.advance_frame();
        self.frame_idx = self.frame_idx.overflowing_add(1).0;

        let width = frame_state.window_cfg.width;
        let height = frame_state.window_cfg.height;

        let frame_constants_offset = self.dynamic_constants.push(FrameConstants {
            view_constants: ViewConstants::builder(frame_state.camera_matrices, width, height)
                .build(),
            mouse: gen_shader_mouse_state(&frame_state),
            frame_idx: self.frame_idx,
        });

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
        let raw_device = &self.backend.device.raw;

        let mut output_img_tracker = LocalImageStateTracker::new(
            self.output_img.raw(),
            vk_sync::AccessType::Nothing,
            cb.raw,
            raw_device,
        );

        let mut sdf_img_tracker = LocalImageStateTracker::new(
            self.sdf_img.raw(),
            if self.frame_idx == 0 {
                vk_sync::AccessType::Nothing
            } else {
                vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer
            },
            cb.raw,
            raw_device,
        );

        unsafe {
            raw_device
                .begin_command_buffer(
                    cb.raw,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            output_img_tracker.transition(vk_sync::AccessType::ComputeShaderWrite);
            sdf_img_tracker.transition(vk_sync::AccessType::ComputeShaderWrite);

            // Edit the SDF
            {
                let shader = self.cs_cache.get(if self.frame_idx == 0 {
                    // Clear if this is the first frame
                    self.gen_empty_sdf
                } else {
                    self.edit_sdf
                });

                self.bind_compute_pipeline(cb, &*shader);
                self.push_descriptor_set(
                    cb,
                    &*shader,
                    0,
                    &[view::image_rw(&self.sdf_img.view(Default::default()))],
                );
                self.bind_frame_constants(cb, &*shader, frame_constants_offset);

                raw_device.cmd_dispatch(cb.raw, SDF_DIM / 4, SDF_DIM / 4, SDF_DIM / 4);
            }

            sdf_img_tracker
                .transition(vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer);

            {
                let shader = self.cs_cache.get(self.clear_bricks_meta);
                self.bind_compute_pipeline(cb, &*shader);
                self.push_descriptor_set(
                    cb,
                    &*shader,
                    0,
                    &[view::buffer_rw(&self.brick_meta_buffer)],
                );
                self.bind_frame_constants(cb, &*shader, frame_constants_offset);
                raw_device.cmd_dispatch(cb.raw, 1, 1, 1);
            }

            {
                let shader = self.cs_cache.get(self.find_sdf_bricks);
                self.bind_compute_pipeline(cb, &*shader);
                self.push_descriptor_set(
                    cb,
                    &*shader,
                    0,
                    &[
                        view::image(&self.sdf_img.view(Default::default())),
                        view::buffer_rw(&self.brick_meta_buffer),
                        view::buffer_rw(&self.brick_inst_buffer),
                    ],
                );
                self.bind_frame_constants(cb, &*shader, frame_constants_offset);
                raw_device.cmd_dispatch(cb.raw, 1, 1, 1);
            }

            // Raymarch the SDF
            {
                let shader = self.cs_cache.get(self.sdf_raymarch_gbuffer);

                self.bind_compute_pipeline(cb, &*shader);
                self.push_descriptor_set(
                    cb,
                    &*shader,
                    0,
                    &[
                        view::image_rw(&self.output_img.view(Default::default())),
                        view::image(&self.sdf_img.view(Default::default())),
                        view::sampler(),
                    ],
                );
                self.bind_frame_constants(cb, &*shader, frame_constants_offset);

                raw_device.cmd_dispatch(cb.raw, (width + 7) / 8, (height + 7) / 8, 1);
            }

            /*{
                output_img_tracker.transition(vk_sync::AccessType::ColorAttachmentWrite);

                self.begin_render_pass(
                    backend,
                    cb,
                    &*self.raster_simple_render_pass,
                    [width, height],
                    &[&self.output_img_view],
                );

                self.set_default_view_and_scissor(backend, cb, [width, height]);

                {
                    raw_device.cmd_bind_pipeline(
                        cb.raw,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.raster_simple.pipeline,
                    );

                    raw_device.cmd_draw(cb.raw, 3, 1, 0, 0);
                }

                raw_device.cmd_end_render_pass(cb.raw);
            }*/

            output_img_tracker
                .transition(vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer);

            blit_image_to_swapchain(
                &*self.backend.device,
                cb,
                &swapchain_image,
                &self.output_img.view(Default::default()),
                &self.present_shader,
            );

            raw_device.end_command_buffer(cb.raw).unwrap();
        }

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

    #[allow(dead_code)]
    fn begin_render_pass(
        &self,
        backend: &RenderBackend,
        cb: &CommandBuffer,
        render_pass: &RenderPass,
        dims: [u32; 2],
        attachments: &[&ImageView],
    ) {
        let framebuffer = render_pass
            .framebuffer_cache
            .get_or_create(
                &backend.device.raw,
                FramebufferCacheKey::new(
                    dims,
                    attachments.iter().copied().map(|a| &a.image.desc),
                    false,
                ),
            )
            .unwrap();

        // Bind images to the imageless framebuffer
        let image_attachments: ArrayVec<[_; MAX_COLOR_ATTACHMENTS]> =
            attachments.iter().copied().map(|a| a.raw).collect();

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
            backend.device.raw.cmd_begin_render_pass(
                cb.raw,
                &pass_begin_desc,
                vk::SubpassContents::INLINE,
            );
        }
    }

    #[allow(dead_code)]
    pub fn set_default_view_and_scissor(
        &self,
        backend: &RenderBackend,
        cb: &CommandBuffer,
        [width, height]: [u32; 2],
    ) {
        unsafe {
            backend.device.raw.cmd_set_viewport(
                cb.raw,
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

            backend.device.raw.cmd_set_scissor(
                cb.raw,
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

    pub fn bind_compute_pipeline(&self, cb: &CommandBuffer, shader: &ComputePipeline) {
        unsafe {
            self.backend.device.raw.cmd_bind_pipeline(
                cb.raw,
                vk::PipelineBindPoint::COMPUTE,
                shader.pipeline,
            );
        }
    }

    pub fn push_descriptor_set(
        &self,
        cb: &CommandBuffer,
        shader: &ComputePipeline,
        set_index: u32,
        bindings: &[DescriptorSetBinding],
    ) {
        let image_info = TempList::new();
        let buffer_info = TempList::new();
        let shader_set_info = &shader.set_layout_info[set_index as usize];

        unsafe {
            let descriptor_writes: Vec<vk::WriteDescriptorSet> = bindings
                .iter()
                .enumerate()
                .filter(|(binding_idx, _)| shader_set_info.contains_key(&(*binding_idx as u32)))
                .map(|(binding_idx, binding)| {
                    let write = vk::WriteDescriptorSet::builder()
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
                        DescriptorSetBinding::Buffer(buffer) => write
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(buffer_info.add(*buffer)))
                            .build(),
                        DescriptorSetBinding::Sampler => {
                            let mut write =
                                write.descriptor_type(vk::DescriptorType::SAMPLER).build();
                            write.descriptor_count = 1;
                            write
                        }
                    }
                })
                .collect();

            self.backend
                .device
                .cmd_ext
                .push_descriptor
                .cmd_push_descriptor_set(
                    cb.raw,
                    vk::PipelineBindPoint::COMPUTE,
                    shader.pipeline_layout,
                    set_index,
                    &descriptor_writes,
                );
        }
    }

    pub fn bind_frame_constants(
        &self,
        cb: &CommandBuffer,
        shader: &ComputePipeline,
        frame_constants_offset: u32,
    ) {
        unsafe {
            self.backend.device.raw.cmd_bind_descriptor_sets(
                cb.raw,
                vk::PipelineBindPoint::COMPUTE,
                shader.pipeline_layout,
                2,
                &[self.frame_descriptor_set],
                &[frame_constants_offset],
            );
        }
    }

    pub fn prepare_frame(&mut self) -> anyhow::Result<()> {
        self.cs_cache.prepare_frame(&self.backend.device)?;

        Ok(())
    }
}

fn gen_shader_mouse_state(frame_state: &FrameState) -> [f32; 4] {
    let pos = frame_state.input.mouse.pos
        / Vec2::new(
            frame_state.window_cfg.width as f32,
            frame_state.window_cfg.height as f32,
        );

    [
        pos.x(),
        pos.y(),
        if (frame_state.input.mouse.button_mask & 1) != 0 {
            1.0
        } else {
            0.0
        },
        if frame_state.input.keys.is_down(VirtualKeyCode::LShift) {
            -1.0
        } else {
            1.0
        },
    ]
}
