// silence unneeded_field_pattern due to offset_of, cast_ptr_alignment in memory management
#![allow(clippy::unneeded_field_pattern, clippy::cast_ptr_alignment)]

use arrayvec::ArrayVec;
use ash::{vk, Device};
use bytemuck::bytes_of;
use egui::{epaint::Vertex, vec2, Context};
use memoffset::offset_of;
use std::{
    ffi::CStr,
    mem,
    os::raw::{c_uchar, c_void},
    slice,
};

pub use egui;

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

#[allow(dead_code)]
fn align_up(x: u32, alignment: u32) -> u32 {
    (x + alignment - 1) & !(alignment - 1)
}

pub struct Renderer {
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
    pipeline_layout: vk::PipelineLayout,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
    pipeline: Option<vk::Pipeline>,
    vertex_buffers: [vk::Buffer; Renderer::FRAME_COUNT],
    vertex_mem_offsets: [usize; Renderer::FRAME_COUNT],
    index_buffers: [vk::Buffer; Renderer::FRAME_COUNT],
    index_mem_offsets: [usize; Renderer::FRAME_COUNT],
    image_buffer: vk::Buffer,
    #[allow(dead_code)]
    host_mem: vk::DeviceMemory,
    host_mapping: *mut c_void,
    image_width: u32,
    image_height: u32,
    image: vk::Image,
    _local_mem: vk::DeviceMemory,
    descriptor_set: vk::DescriptorSet,
    #[allow(dead_code)]
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

    const INDEX_BUFFER_SIZE: usize = Renderer::INDEX_COUNT_PER_FRAME * mem::size_of::<u32>();
    const VERTEX_BUFFER_SIZE: usize = Renderer::VERTEX_COUNT_PER_FRAME * mem::size_of::<Vertex>();

    pub fn new(
        physical_width: u32,
        physical_height: u32,
        scale_factor: f64,
        device: &Device,
        physical_device_properties: &vk::PhysicalDeviceProperties,
        physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        egui: &mut Context,
    ) -> Self {
        let vertex_shader = load_shader_module(device, include_bytes!("egui.vert.spv"));
        let fragment_shader = load_shader_module(device, include_bytes!("egui.frag.spv"));

        let sampler = {
            let sampler_create_info = vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(false)
                .min_filter(vk::Filter::NEAREST)
                .mag_filter(vk::Filter::NEAREST)
                .min_lod(0.0)
                .max_lod(vk::LOD_CLAMP_NONE);
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
                size: (Renderer::VERTEX_COUNT_PER_FRAME * mem::size_of::<Vertex>())
                    as vk::DeviceSize,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
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
                size: (Renderer::INDEX_COUNT_PER_FRAME * mem::size_of::<u32>()) as vk::DeviceSize,
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

        let full_output = egui.run(
            egui::RawInput {
                pixels_per_point: Some(scale_factor as f32),
                screen_rect: Some(egui::Rect::from_min_size(
                    Default::default(),
                    vec2(physical_width as f32, physical_height as f32) / scale_factor as f32,
                )),
                time: Some(0.0),
                ..Default::default()
            },
            |_ctx| {},
        );
        let texture_size = egui.fonts().font_image_size();
        let texture_delta = full_output.textures_delta.set.iter().next().unwrap();
        let texture = match &texture_delta.1.image {
            egui::ImageData::Alpha(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );

                image
            }
            _ => panic!("Egui font texture could not be loaded"),
        };

        let (image_buffer, image_mem_offset) = {
            let buffer_create_info = vk::BufferCreateInfo {
                size: vk::DeviceSize::from(texture_size[0] as u64 * texture_size[1] as u64 * 4),
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
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
            let image_create_info = vk::ImageCreateInfo::builder()
                .format(vk::Format::R8G8B8A8_UNORM)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .image_type(vk::ImageType::TYPE_2D)
                .mip_levels(1)
                .array_layers(1)
                .extent(vk::Extent3D {
                    width: texture_size[0] as u32,
                    height: texture_size[1] as u32,
                    depth: 1,
                });
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
            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .format(vk::Format::R8G8B8A8_SRGB)
                .view_type(vk::ImageViewType::TYPE_2D)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .base_mip_level(0)
                        .layer_count(1)
                        .level_count(1)
                        .build(),
                );
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

            let srgba_pixels: Vec<u8> = texture
                .srgba_pixels(0.5)
                .flat_map(|srgba| vec![srgba.r(), srgba.g(), srgba.b(), srgba.a()])
                .collect();
            unsafe {
                image_base.copy_from_nonoverlapping(srgba_pixels.as_ptr(), srgba_pixels.len())
            };
        }

        Self {
            physical_width,
            physical_height,
            scale_factor,
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
            image_width: texture_size[0] as u32,
            image_height: texture_size[1] as u32,
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
                stride: mem::size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            };
            let vertex_input_attributes = [
                // position
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(offset_of!(Vertex, pos) as u32)
                    .location(0)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // uv
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(offset_of!(Vertex, uv) as u32)
                    .location(1)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // color
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(offset_of!(Vertex, color) as u32)
                    .location(2)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .build(),
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

            let rasterization_state_create_info =
                vk::PipelineRasterizationStateCreateInfo::builder()
                    .depth_clamp_enable(false)
                    .rasterizer_discard_enable(false)
                    .polygon_mode(vk::PolygonMode::FILL)
                    .cull_mode(vk::CullModeFlags::NONE)
                    .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                    .depth_bias_enable(false)
                    .line_width(1.0);

            let stencil_op = vk::StencilOpState::builder()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .build();
            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(false)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::ALWAYS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .front(stencil_op)
                .back(stencil_op);

            let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            let color_blend_attachments = [vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::all(),
            }];

            let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(&color_blend_attachments);

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
                .depth_stencil_state(&depth_stencil_info)
                .color_blend_state(&color_blend_info)
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
        clipped_meshes: Vec<egui::ClippedMesh>,
        device: &Device,
        command_buffer: vk::CommandBuffer,
    ) {
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
            }

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
                    vk::IndexType::UINT32,
                );
            }

            unsafe {
                device.cmd_set_viewport(
                    command_buffer,
                    0,
                    &[vk::Viewport::builder()
                        .x(0.0)
                        .y(0.0)
                        .width(self.physical_width as f32)
                        .height(self.physical_height as f32)
                        .min_depth(0.0)
                        .max_depth(1.0)
                        .build()],
                );
            };

            let width_points = self.physical_width as f32 / self.scale_factor as f32;
            let height_points = self.physical_height as f32 / self.scale_factor as f32;

            unsafe {
                device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytes_of(&width_points),
                );
                device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    std::mem::size_of_val(&width_points) as u32,
                    bytes_of(&height_points),
                );
            };

            // let clip_off = draw_data.display_pos;
            // let clip_scale = draw_data.framebuffer_scale;
            let vertex_base =
                unsafe { (self.host_mapping as *mut u8).add(vertex_mem_offset) } as *mut Vertex;
            let index_base =
                unsafe { (self.host_mapping as *mut u8).add(index_mem_offset) } as *mut u32;
            let mut vertex_offset = 0;
            let mut index_offset = 0;
            for egui::ClippedMesh(rect, mesh) in clipped_meshes {
                // update texture
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        slice::from_ref(&self.descriptor_set),
                        &[],
                    );
                }

                if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                    continue;
                }

                let next_vertex_offset = vertex_offset + mesh.vertices.len();
                let next_index_offset = index_offset + mesh.indices.len();
                if next_vertex_offset >= Renderer::VERTEX_BUFFER_SIZE
                    || next_index_offset >= Renderer::INDEX_BUFFER_SIZE
                {
                    break;
                }

                unsafe {
                    vertex_base
                        .add(vertex_offset)
                        .copy_from_nonoverlapping(mesh.vertices.as_ptr(), mesh.vertices.len());
                    index_base
                        .add(index_offset)
                        .copy_from_nonoverlapping(mesh.indices.as_ptr(), mesh.indices.len());
                }

                // record draw commands
                unsafe {
                    let min = rect.min;
                    let min = egui::Pos2 {
                        x: min.x * self.scale_factor as f32,
                        y: min.y * self.scale_factor as f32,
                    };
                    let min = egui::Pos2 {
                        x: f32::clamp(min.x, 0.0, self.physical_width as f32),
                        y: f32::clamp(min.y, 0.0, self.physical_height as f32),
                    };
                    let max = rect.max;
                    let max = egui::Pos2 {
                        x: max.x * self.scale_factor as f32,
                        y: max.y * self.scale_factor as f32,
                    };
                    let max = egui::Pos2 {
                        x: f32::clamp(max.x, min.x, self.physical_width as f32),
                        y: f32::clamp(max.y, min.y, self.physical_height as f32),
                    };
                    device.cmd_set_scissor(
                        command_buffer,
                        0,
                        &[vk::Rect2D::builder()
                            .offset(
                                vk::Offset2D::builder()
                                    .x(min.x.round() as i32)
                                    .y(min.y.round() as i32)
                                    .build(),
                            )
                            .extent(
                                vk::Extent2D::builder()
                                    .width((max.x.round() - min.x) as u32)
                                    .height((max.y.round() - min.y) as u32)
                                    .build(),
                            )
                            .build()],
                    );
                    device.cmd_draw_indexed(
                        command_buffer,
                        mesh.indices.len() as u32,
                        1,
                        index_offset as u32,
                        vertex_offset as i32,
                        0,
                    );
                    index_offset += mesh.indices.len();
                }

                vertex_offset = next_vertex_offset;
                assert_eq!(index_offset, next_index_offset);
            }
        }
    }
}
