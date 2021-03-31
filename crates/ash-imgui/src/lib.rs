// silence unneeded_field_pattern due to offset_of, cast_ptr_alignment in memory management
#![allow(clippy::unneeded_field_pattern, clippy::cast_ptr_alignment)]

use arrayvec::ArrayVec;
use ash::version::DeviceV1_0;
use ash::{vk, Device};
use imgui::internal::RawWrapper;
use imgui::{Context, DrawCmd, DrawCmdParams, DrawData, DrawIdx, DrawVert};
use memoffset::offset_of;
use std::ffi::CStr;
use std::mem;
use std::os::raw::{c_uchar, c_void};
use std::slice;

fn load_shader_module(device: &Device, bytes: &[u8]) -> vk::ShaderModule {
    let shader_module_create_info = vk::ShaderModuleCreateInfo {
        code_size: bytes.len(),
        p_code: bytes.as_ptr() as *const u32,
        ..Default::default()
    };
    unsafe { device.create_shader_module(&shader_module_create_info, None) }.unwrap()
}

fn get_memory_type_index(
    physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    property_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..physical_device_memory_properties.memory_type_count {
        let mt = &physical_device_memory_properties.memory_types[i as usize];
        if (type_filter & (1 << i)) != 0 && mt.property_flags.contains(property_flags) {
            return Some(i);
        }
    }
    None
}

fn align_up(x: u32, alignment: u32) -> u32 {
    (x + alignment - 1) & !(alignment - 1)
}

pub struct Renderer {
    pipeline_layout: vk::PipelineLayout,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
    pipeline: Option<vk::Pipeline>,
    vertex_buffers: [vk::Buffer; Renderer::FRAME_COUNT],
    vertex_mem_offsets: [usize; Renderer::FRAME_COUNT],
    index_buffers: [vk::Buffer; Renderer::FRAME_COUNT],
    index_mem_offsets: [usize; Renderer::FRAME_COUNT],
    image_buffer: vk::Buffer,
    host_mem: vk::DeviceMemory,
    host_mapping: *mut c_void,
    image_width: u32,
    image_height: u32,
    image: vk::Image,
    _local_mem: vk::DeviceMemory,
    descriptor_set: vk::DescriptorSet,
    atom_size: u32,
    frame_index: usize,
    image_needs_copy: bool,
}

impl Renderer {
    const QUAD_COUNT_PER_FRAME: usize = 64 * 1024;
    const VERTEX_COUNT_PER_FRAME: usize = 4 * Renderer::QUAD_COUNT_PER_FRAME;
    const INDEX_COUNT_PER_FRAME: usize = 6 * Renderer::QUAD_COUNT_PER_FRAME;
    const PUSH_CONSTANT_SIZE: usize = 8;
    const FRAME_COUNT: usize = 2;

    pub fn new(
        device: &Device,
        physical_device_properties: &vk::PhysicalDeviceProperties,
        physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        imgui: &mut Context,
    ) -> Self {
        let vertex_shader = load_shader_module(device, include_bytes!("imgui.vert.spv"));
        let fragment_shader = load_shader_module(device, include_bytes!("imgui.frag.spv"));

        let sampler = {
            let sampler_create_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                ..Default::default()
            };
            unsafe { device.create_sampler(&sampler_create_info, None) }.unwrap()
        };

        let descriptor_set_layout = {
            let binding = vk::DescriptorSetLayoutBinding::builder()
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(slice::from_ref(&sampler));
            let descriptor_set_layout_create_info =
                vk::DescriptorSetLayoutCreateInfo::builder().bindings(slice::from_ref(&binding));
            unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None) }
                .unwrap()
        };

        let pipeline_layout = {
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: Renderer::PUSH_CONSTANT_SIZE as u32,
            };
            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(slice::from_ref(&descriptor_set_layout))
                .push_constant_ranges(slice::from_ref(&push_constant_range));
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }.unwrap()
        };

        let mut host_allocation_size = 0;
        let mut host_memory_type_filter = 0xffff_ffff;

        let (vertex_buffers, vertex_mem_offsets) = {
            let buffer_create_info = vk::BufferCreateInfo {
                size: (Renderer::VERTEX_COUNT_PER_FRAME * mem::size_of::<DrawVert>())
                    as vk::DeviceSize,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                ..Default::default()
            };
            let mut buffers = ArrayVec::<[vk::Buffer; Renderer::FRAME_COUNT]>::new();
            let mut mem_offsets = ArrayVec::<[usize; Renderer::FRAME_COUNT]>::new();
            for _i in 0..Renderer::FRAME_COUNT {
                let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }.unwrap();
                let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
                assert_eq!(mem_req.size, buffer_create_info.size);
                let mem_offset = host_allocation_size as usize;
                host_allocation_size += buffer_create_info.size;
                buffers.push(buffer);
                mem_offsets.push(mem_offset);
                host_memory_type_filter &= mem_req.memory_type_bits;
            }
            (
                buffers.into_inner().unwrap(),
                mem_offsets.into_inner().unwrap(),
            )
        };

        let (index_buffers, index_mem_offsets) = {
            let buffer_create_info = vk::BufferCreateInfo {
                size: (Renderer::INDEX_COUNT_PER_FRAME * mem::size_of::<DrawIdx>())
                    as vk::DeviceSize,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                ..Default::default()
            };
            let mut buffers = ArrayVec::<[vk::Buffer; Renderer::FRAME_COUNT]>::new();
            let mut mem_offsets = ArrayVec::<[usize; Renderer::FRAME_COUNT]>::new();
            for _i in 0..Renderer::FRAME_COUNT {
                let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }.unwrap();
                let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
                assert_eq!(mem_req.size, buffer_create_info.size);
                let mem_offset = host_allocation_size as usize;
                host_allocation_size += buffer_create_info.size;
                buffers.push(buffer);
                mem_offsets.push(mem_offset);
                host_memory_type_filter &= mem_req.memory_type_bits;
            }
            (
                buffers.into_inner().unwrap(),
                mem_offsets.into_inner().unwrap(),
            )
        };

        let mut fonts = imgui.fonts();
        let texture = fonts.build_alpha8_texture();

        let (image_buffer, image_mem_offset) = {
            let buffer_create_info = vk::BufferCreateInfo {
                size: vk::DeviceSize::from(texture.width * texture.height),
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                ..Default::default()
            };
            let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }.unwrap();
            let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
            assert_eq!(mem_req.size, buffer_create_info.size);
            let mem_offset = host_allocation_size as usize;
            host_allocation_size += buffer_create_info.size;
            host_memory_type_filter &= mem_req.memory_type_bits;
            (buffer, mem_offset)
        };

        let host_mem = {
            let memory_type_index = get_memory_type_index(
                physical_device_memory_properties,
                host_memory_type_filter,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            )
            .unwrap();
            let memory_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: host_allocation_size,
                memory_type_index,
                ..Default::default()
            };
            unsafe { device.allocate_memory(&memory_allocate_info, None) }.unwrap()
        };

        for (&buf, &ofs) in vertex_buffers.iter().zip(vertex_mem_offsets.iter()) {
            unsafe { device.bind_buffer_memory(buf, host_mem, ofs as vk::DeviceSize) }.unwrap();
        }
        for (&buf, &ofs) in index_buffers.iter().zip(index_mem_offsets.iter()) {
            unsafe { device.bind_buffer_memory(buf, host_mem, ofs as vk::DeviceSize) }.unwrap();
        }
        unsafe {
            device.bind_buffer_memory(image_buffer, host_mem, image_mem_offset as vk::DeviceSize)
        }
        .unwrap();

        let host_mapping =
            unsafe { device.map_memory(host_mem, 0, vk::WHOLE_SIZE, Default::default()) }.unwrap();

        let image = {
            let image_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8_UNORM,
                extent: vk::Extent3D {
                    width: texture.width,
                    height: texture.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                ..Default::default()
            };
            unsafe { device.create_image(&image_create_info, None) }.unwrap()
        };

        let (local_allocation_size, local_memory_type_filter) = {
            let mem_req = unsafe { device.get_image_memory_requirements(image) };
            (mem_req.size, mem_req.memory_type_bits)
        };

        let local_mem = {
            let memory_type_index = get_memory_type_index(
                physical_device_memory_properties,
                local_memory_type_filter,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .unwrap();
            let memory_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: local_allocation_size,
                memory_type_index,
                ..Default::default()
            };
            unsafe { device.allocate_memory(&memory_allocate_info, None) }.unwrap()
        };

        unsafe { device.bind_image_memory(image, local_mem, 0) }.unwrap();

        let descriptor_pool = {
            let descriptor_pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            }];
            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&descriptor_pool_sizes);
            unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }.unwrap()
        };

        let descriptor_set = {
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(slice::from_ref(&descriptor_set_layout));
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }.unwrap()[0]
        };

        let image_view = {
            let image_view_create_info = vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: vk::Format::R8_UNORM,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::ONE,
                    g: vk::ComponentSwizzle::ONE,
                    b: vk::ComponentSwizzle::ONE,
                    a: vk::ComponentSwizzle::R,
                },
                ..Default::default()
            };
            unsafe { device.create_image_view(&image_view_create_info, None) }.unwrap()
        };

        {
            let image_info = vk::DescriptorImageInfo {
                sampler,
                image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            let write_descriptor_set = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(slice::from_ref(&image_info));
            unsafe { device.update_descriptor_sets(slice::from_ref(&write_descriptor_set), &[]) };
        }

        let atom_size = physical_device_properties.limits.non_coherent_atom_size as u32;

        {
            let image_base =
                unsafe { (host_mapping as *mut u8).add(image_mem_offset) } as *mut c_uchar;

            assert_eq!(texture.data.len() as u32, texture.width * texture.height);
            unsafe {
                image_base.copy_from_nonoverlapping(texture.data.as_ptr(), texture.data.len())
            };

            let mapped_memory_range = vk::MappedMemoryRange {
                memory: host_mem,
                offset: image_mem_offset as vk::DeviceSize,
                size: vk::DeviceSize::from(align_up(texture.data.len() as u32, atom_size)),
                ..Default::default()
            };
            unsafe { device.flush_mapped_memory_ranges(slice::from_ref(&mapped_memory_range)) }
                .unwrap();
        }

        Self {
            pipeline_layout,
            vertex_shader,
            fragment_shader,
            pipeline: None,
            vertex_buffers,
            vertex_mem_offsets,
            index_buffers,
            index_mem_offsets,
            image_buffer,
            host_mem,
            host_mapping,
            image_width: texture.width,
            image_height: texture.height,
            image,
            _local_mem: local_mem,
            descriptor_set,
            atom_size,
            frame_index: 0,
            image_needs_copy: true,
        }
    }

    pub fn begin_frame(&mut self, device: &Device, command_buffer: vk::CommandBuffer) {
        self.frame_index = (1 + self.frame_index) % Renderer::FRAME_COUNT;

        if self.image_needs_copy {
            let transfer_from_undef = vk::ImageMemoryBarrier {
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    slice::from_ref(&transfer_from_undef),
                )
            };

            let buffer_image_copy = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                image_extent: vk::Extent3D {
                    width: self.image_width,
                    height: self.image_height,
                    depth: 1,
                },
                ..Default::default()
            };
            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    self.image_buffer,
                    self.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    slice::from_ref(&buffer_image_copy),
                )
            };

            let shader_from_transfer = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    slice::from_ref(&shader_from_transfer),
                )
            };

            self.image_needs_copy = false;
        }
    }

    pub fn has_pipeline(&self) -> bool {
        self.pipeline.is_some()
    }

    pub fn create_pipeline(
        &mut self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Option<vk::Pipeline> {
        let pipeline = {
            let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
            let shader_stage_create_info = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: self.vertex_shader,
                    p_name: shader_entry_name.as_ptr(),
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: self.fragment_shader,
                    p_name: shader_entry_name.as_ptr(),
                    ..Default::default()
                },
            ];

            let vertex_input_binding = vk::VertexInputBindingDescription {
                binding: 0,
                stride: mem::size_of::<DrawVert>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            };
            let vertex_input_attributes = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: offset_of!(DrawVert, pos) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: offset_of!(DrawVert, uv) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: vk::Format::R8G8B8A8_UNORM,
                    offset: offset_of!(DrawVert, col) as u32,
                },
            ];

            let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(slice::from_ref(&vertex_input_binding))
                .vertex_attribute_descriptions(&vertex_input_attributes);

            let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };

            let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            };

            let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            };
            let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::all(),
            };
            let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(slice::from_ref(&color_blend_attachment_state));

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let pipeline_dynamic_state_create_info =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

            let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
                .stages(&shader_stage_create_info)
                .vertex_input_state(&vertex_input_state_create_info)
                .input_assembly_state(&input_assembly_state_create_info)
                .viewport_state(&viewport_state_create_info)
                .rasterization_state(&rasterization_state_create_info)
                .multisample_state(&multisample_state_create_info)
                .color_blend_state(&color_blend_state_create_info)
                .dynamic_state(&pipeline_dynamic_state_create_info)
                .layout(self.pipeline_layout)
                .render_pass(render_pass);

            unsafe {
                device.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    slice::from_ref(&pipeline_create_info),
                    None,
                )
            }
            .unwrap()[0]
        };
        self.pipeline.replace(pipeline)
    }

    pub fn destroy_pipeline(&mut self, device: &Device) {
        let pipeline = self.pipeline.take().unwrap();
        unsafe {
            device.destroy_pipeline(pipeline, None);
        }
    }

    pub fn render(
        &mut self,
        draw_data: &DrawData,
        device: &Device,
        command_buffer: vk::CommandBuffer,
    ) {
        let width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];

        {
            let vertex_buffer = self.vertex_buffers[self.frame_index];
            let vertex_mem_offset = self.vertex_mem_offsets[self.frame_index];
            let index_buffer = self.index_buffers[self.frame_index];
            let index_mem_offset = self.index_mem_offsets[self.frame_index];

            unsafe {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.unwrap(),
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    slice::from_ref(&self.descriptor_set),
                    &[],
                );
            }

            let dims_rcp = [1.0 / width, 1.0 / height];
            unsafe {
                device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(
                        dims_rcp.as_ptr() as *const f32 as *const u8,
                        Renderer::PUSH_CONSTANT_SIZE,
                    ),
                )
            };

            let viewport = vk::Viewport {
                width,
                height,
                max_depth: 1.0,
                ..Default::default()
            };
            unsafe { device.cmd_set_viewport(command_buffer, 0, slice::from_ref(&viewport)) };

            unsafe {
                device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    slice::from_ref(&vertex_buffer),
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    command_buffer,
                    index_buffer,
                    0,
                    vk::IndexType::UINT16,
                );
            }

            let clip_off = draw_data.display_pos;
            let clip_scale = draw_data.framebuffer_scale;
            let vertex_base =
                unsafe { (self.host_mapping as *mut u8).add(vertex_mem_offset) } as *mut DrawVert;
            let index_base =
                unsafe { (self.host_mapping as *mut u8).add(index_mem_offset) } as *mut DrawIdx;
            let mut vertex_offset = 0;
            let mut index_offset = 0;
            for draw_list in draw_data.draw_lists() {
                let vtx_buffer = draw_list.vtx_buffer();
                let idx_buffer = draw_list.idx_buffer();
                let next_vertex_offset = vertex_offset + vtx_buffer.len();
                let next_index_offset = index_offset + idx_buffer.len();
                if next_vertex_offset > Renderer::VERTEX_COUNT_PER_FRAME
                    || next_index_offset > Renderer::INDEX_COUNT_PER_FRAME
                {
                    break;
                }

                unsafe {
                    vertex_base
                        .add(vertex_offset)
                        .copy_from_nonoverlapping(vtx_buffer.as_ptr(), vtx_buffer.len());
                    index_base
                        .add(index_offset)
                        .copy_from_nonoverlapping(idx_buffer.as_ptr(), idx_buffer.len());
                }

                for cmd in draw_list.commands() {
                    match cmd {
                        DrawCmd::Elements {
                            count,
                            cmd_params: DrawCmdParams { clip_rect, .. },
                        } => {
                            let clip_rect = [
                                (clip_rect[0] - clip_off[0]) * clip_scale[0],
                                (clip_rect[1] - clip_off[1]) * clip_scale[1],
                                (clip_rect[2] - clip_off[0]) * clip_scale[0],
                                (clip_rect[3] - clip_off[1]) * clip_scale[1],
                            ];
                            let scissor = vk::Rect2D {
                                offset: vk::Offset2D {
                                    x: clip_rect[0].floor() as i32,
                                    y: clip_rect[1].floor() as i32,
                                },
                                extent: vk::Extent2D {
                                    width: (clip_rect[2] - clip_rect[0]).ceil() as u32,
                                    height: (clip_rect[3] - clip_rect[1]).ceil() as u32,
                                },
                            };
                            let count = count as u32;
                            unsafe {
                                device.cmd_set_scissor(
                                    command_buffer,
                                    0,
                                    slice::from_ref(&scissor),
                                );
                                device.cmd_draw_indexed(
                                    command_buffer,
                                    count,
                                    1,
                                    index_offset as u32,
                                    vertex_offset as i32,
                                    0,
                                );
                            }
                            index_offset += count as usize;
                        }
                        DrawCmd::ResetRenderState => {}
                        DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                            callback(draw_list.raw(), raw_cmd)
                        },
                    }
                }

                vertex_offset = next_vertex_offset;
                assert_eq!(index_offset, next_index_offset);
            }

            let mapped_ranges = [
                vk::MappedMemoryRange {
                    memory: self.host_mem,
                    offset: vertex_mem_offset as vk::DeviceSize,
                    size: vk::DeviceSize::from(align_up(
                        (vertex_offset * mem::size_of::<DrawVert>()) as u32,
                        self.atom_size,
                    )),
                    ..Default::default()
                },
                vk::MappedMemoryRange {
                    memory: self.host_mem,
                    offset: index_mem_offset as vk::DeviceSize,
                    size: vk::DeviceSize::from(align_up(
                        (index_offset * mem::size_of::<DrawIdx>()) as u32,
                        self.atom_size,
                    )),
                    ..Default::default()
                },
            ];
            unsafe { device.flush_mapped_memory_ranges(&mapped_ranges) }.unwrap();
        }
    }
}
