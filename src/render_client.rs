use crate::{
    asset::{image::RawRgba8Image, mesh::PackedTriangleMesh},
    backend::{self, image::*, shader::*, RenderBackend},
    dynamic_constants::DynamicConstants,
    render_passes::{RasterMeshesData, UploadedTriMesh},
    renderer::*,
    rg,
    rg::RetiredRenderGraph,
    viewport::ViewConstants,
    FrameState,
};
use backend::buffer::{Buffer, BufferDesc};
use byte_slice_cast::AsByteSlice;
use glam::Vec2;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use slingshot::{
    ash::{version::DeviceV1_0, vk},
    backend::device,
    vk_sync,
};
use std::{mem::size_of, sync::Arc};
use winit::VirtualKeyCode;

#[repr(C)]
#[derive(Copy, Clone)]
struct FrameConstants {
    view_constants: ViewConstants,
    mouse: [f32; 4],
    frame_idx: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct GpuMesh {
    vertex_core_offset: u32,
    vertex_uv_offset: u32,
    vertex_mat_offset: u32,
    vertex_aux_offset: u32,
    mat_data_offset: u32,
}

const MAX_GPU_MESHES: usize = 1024;
const VERTEX_BUFFER_CAPACITY: usize = 1024 * 1024 * 128;

pub struct VickiRenderClient {
    device: Arc<device::Device>,
    raster_simple_render_pass: Arc<RenderPass>,
    //sdf_img: TemporalImage,
    //cube_index_buffer: Arc<Buffer>,
    meshes: Vec<UploadedTriMesh>,
    mesh_buffer: Arc<Buffer>,
    vertex_buffer: Arc<Buffer>,
    vertex_buffer_size: usize,
    bindless_descriptor_set: vk::DescriptorSet,
    bindless_images: Vec<Image>,
    frame_idx: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BindlessImageHandle(pub u32);

/*fn as_byte_slice_unchecked<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * size_of::<T>()) }
}*/

fn append_buffer_data<T: Copy>(
    buf_ptr: *mut u8,
    buf_size: &mut usize,
    buf_capacity: usize,
    data: &[T],
) -> usize {
    if !data.is_empty() {
        let alignment = std::mem::align_of::<T>();
        assert!(alignment.count_ones() == 1);

        let data_start = (*buf_size + alignment - 1) & !(alignment - 1);
        let data_bytes = data.len() * size_of::<T>();
        assert!(data_start + data_bytes <= buf_capacity);

        let dst = unsafe {
            std::slice::from_raw_parts_mut(buf_ptr.add(data_start) as *mut T, data.len())
        };
        dst.copy_from_slice(data);

        *buf_size = data_start + data_bytes;
        data_start
    } else {
        0
    }
}

fn create_bindless_descriptor_set(device: &device::Device) -> vk::DescriptorSet {
    let raw_device = &device.raw;

    let set_binding_flags = [vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
        | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
        | vk::DescriptorBindingFlags::PARTIALLY_BOUND
        | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT];
    let mut binding_flags_create_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
        .binding_flags(&set_binding_flags)
        .build();

    let descriptor_set_layout = unsafe {
        raw_device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&[vk::DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_count(MAX_BINDLESS_DESCRIPTOR_COUNT as _)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .stage_flags(
                            vk::ShaderStageFlags::COMPUTE
                                | vk::ShaderStageFlags::ALL_GRAPHICS
                                | vk::ShaderStageFlags::RAYGEN_KHR,
                        )
                        .build()])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags_create_info)
                    .build(),
                None,
            )
            .unwrap()
    };

    let descriptor_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_BINDLESS_DESCRIPTOR_COUNT as _,
    }];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&descriptor_sizes)
        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
        .max_sets(1);

    let descriptor_pool = unsafe {
        raw_device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap()
    };

    let set = unsafe {
        raw_device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                    .build(),
            )
            .unwrap()[0]
    };

    set
}

struct BufferBuilder<'a> {
    buf_ptr: *mut u8,
    buf_size: &'a mut usize,
    buf_capacity: usize,
}

impl<'a> BufferBuilder<'a> {
    fn new(buf_ptr: *mut u8, buf_size: &'a mut usize, buf_capacity: usize) -> Self {
        Self {
            buf_ptr,
            buf_size,
            buf_capacity,
        }
    }

    fn append<T: Copy>(&mut self, data: &[T]) -> usize {
        append_buffer_data(self.buf_ptr, &mut self.buf_size, self.buf_capacity, data)
    }
}

impl VickiRenderClient {
    pub fn new(backend: &RenderBackend) -> anyhow::Result<Self> {
        /*let cube_indices = cube_indices();
        let cube_index_buffer = backend.device.create_buffer(
            BufferDesc {
                size: cube_indices.len() * 4,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
            },
            Some((&cube_indices).as_byte_slice()),
        )?;*/

        let raster_simple_render_pass = create_render_pass(
            &*backend.device,
            RenderPassDesc {
                color_attachments: &[RenderPassAttachmentDesc::new(
                    vk::Format::R16G16B16A16_SFLOAT,
                )
                .garbage_input()],
                depth_attachment: Some(RenderPassAttachmentDesc::new(
                    vk::Format::D24_UNORM_S8_UINT,
                )),
            },
        )?;

        let mesh_buffer = Arc::new(
            backend
                .device
                .create_buffer(
                    BufferDesc {
                        size: MAX_GPU_MESHES * size_of::<GpuMesh>(),
                        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                        mapped: true,
                    },
                    None,
                )
                .unwrap(),
        );

        let vertex_buffer = Arc::new(
            backend
                .device
                .create_buffer(
                    BufferDesc {
                        size: VERTEX_BUFFER_CAPACITY,
                        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                        mapped: true,
                    },
                    None,
                )
                .unwrap(),
        );

        let bindless_descriptor_set = create_bindless_descriptor_set(backend.device.as_ref());

        Ok(Self {
            raster_simple_render_pass,

            //sdf_img: TemporalImage::new(Arc::new(sdf_img)),
            //cube_index_buffer: Arc::new(cube_index_buffer),
            device: backend.device.clone(),
            meshes: Default::default(),
            mesh_buffer,
            vertex_buffer,
            vertex_buffer_size: 0,
            bindless_descriptor_set,
            bindless_images: Default::default(),
            frame_idx: 0u32,
        })
    }

    pub fn add_image(&mut self, src: &RawRgba8Image) -> BindlessImageHandle {
        let image = self
            .device
            .create_image(
                ImageDesc::new_2d(vk::Format::R8G8B8A8_SRGB, src.dimensions)
                    .usage(vk::ImageUsageFlags::SAMPLED),
                Some(ImageSubResourceData {
                    data: &src.data,
                    row_pitch: src.dimensions[0] as usize * 4,
                    slice_pitch: 0,
                }),
            )
            .unwrap();

        let handle = BindlessImageHandle(self.bindless_images.len() as _);
        let view = image.view(self.device.as_ref(), &ImageViewDesc::default());

        self.bindless_images.push(image);

        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .build();

        let write_descriptor_set = vk::WriteDescriptorSet::builder()
            .dst_set(self.bindless_descriptor_set)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_binding(0)
            .dst_array_element(handle.0 as _)
            .image_info(std::slice::from_ref(&image_info))
            .build();

        unsafe {
            self.device
                .raw
                .update_descriptor_sets(std::slice::from_ref(&write_descriptor_set), &[]);
        }

        handle
    }

    pub fn add_mesh(&mut self, mesh: PackedTriangleMesh) {
        let mesh_idx = self.meshes.len();

        let index_buffer = Arc::new(
            self.device
                .create_buffer(
                    BufferDesc {
                        size: mesh.indices.len() * 4,
                        usage: vk::BufferUsageFlags::INDEX_BUFFER,
                        mapped: false,
                    },
                    Some((&mesh.indices).as_byte_slice()),
                )
                .unwrap(),
        );

        let mut buffer_builder = BufferBuilder::new(
            self.vertex_buffer.allocation_info.get_mapped_data(),
            &mut self.vertex_buffer_size,
            VERTEX_BUFFER_CAPACITY,
        );

        let vertex_core_offset = buffer_builder.append(&mesh.verts) as _;
        let vertex_uv_offset = buffer_builder.append(&mesh.uvs) as _;
        let vertex_mat_offset = buffer_builder.append(&mesh.material_ids) as _;
        let vertex_aux_offset = buffer_builder.append(&mesh.colors) as _;
        let mat_data_offset = buffer_builder.append(&mesh.materials) as _;

        let mesh_buffer_dst = unsafe {
            let mesh_buffer_dst =
                self.mesh_buffer.allocation_info.get_mapped_data() as *mut GpuMesh;
            assert!(!mesh_buffer_dst.is_null());
            std::slice::from_raw_parts_mut(mesh_buffer_dst, MAX_GPU_MESHES)
        };

        mesh_buffer_dst[mesh_idx] = GpuMesh {
            vertex_core_offset,
            vertex_uv_offset,
            vertex_mat_offset,
            vertex_aux_offset,
            mat_data_offset,
        };

        self.meshes.push(UploadedTriMesh {
            index_buffer,
            index_count: mesh.indices.len() as _,
        });
    }
}

impl RenderClient<FrameState> for VickiRenderClient {
    fn prepare_render_graph(
        &mut self,
        rg: &mut crate::rg::RenderGraph,
        frame_state: &FrameState,
    ) -> rg::ExportedHandle<Image> {
        /*let mut sdf_img = rg.import_image(self.sdf_img.resource.clone(), self.sdf_img.access_type);
        let cube_index_buffer = rg.import_buffer(
            self.cube_index_buffer.clone(),
            vk_sync::AccessType::TransferWrite,
        );*/

        let mut depth_img = crate::render_passes::create_image(
            rg,
            ImageDesc::new_2d(vk::Format::D24_UNORM_S8_UINT, frame_state.window_cfg.dims()),
        );
        crate::render_passes::clear_depth(rg, &mut depth_img);
        /*crate::render_passes::edit_sdf(rg, &mut sdf_img, self.frame_idx == 0);

        let sdf_raster_bricks: SdfRasterBricks =
            crate::render_passes::calculate_sdf_bricks_meta(rg, &sdf_img);*/
        /*let mut tex = crate::render_passes::raymarch_sdf(
            rg,
            &sdf_img,
            ImageDesc::new_2d(
                vk::Format::R16G16B16A16_SFLOAT,
                frame_state.window_cfg.dims(),
            ),
        );*/

        let mut tex = crate::render_passes::create_image(
            rg,
            ImageDesc::new_2d(
                vk::Format::R16G16B16A16_SFLOAT,
                frame_state.window_cfg.dims(),
            ),
        );
        crate::render_passes::clear_color(rg, &mut tex, [0.1, 0.2, 0.5, 1.0]);

        let mesh_buffer = rg.import_buffer(
            self.mesh_buffer.clone(),
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        let vertex_buffer = rg.import_buffer(
            self.vertex_buffer.clone(),
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        crate::render_passes::raster_meshes(
            rg,
            self.raster_simple_render_pass.clone(),
            &mut depth_img,
            &mut tex,
            RasterMeshesData {
                meshes: self.meshes.as_slice(),
                mesh_buffer: &mesh_buffer,
                vertex_buffer: &vertex_buffer,
                bindless_descriptor_set: self.bindless_descriptor_set,
            },
        );

        /*crate::render_passes::raster_sdf(
            rg,
            self.raster_simple_render_pass.clone(),
            &mut depth_img,
            &mut tex,
            crate::render_passes::RasterSdfData {
                sdf_img: &sdf_img,
                brick_inst_buffer: &sdf_raster_bricks.brick_inst_buffer,
                brick_meta_buffer: &sdf_raster_bricks.brick_meta_buffer,
                cube_index_buffer: &cube_index_buffer,
            },
        );*/

        let tex = crate::render_passes::blur(rg, &tex);
        //self.sdf_img.last_rg_handle = Some(rg.export_image(sdf_img, vk::ImageUsageFlags::empty()));

        rg.export_image(tex, vk::ImageUsageFlags::SAMPLED)
    }

    fn prepare_frame_constants(
        &mut self,
        dynamic_constants: &mut DynamicConstants,
        frame_state: &FrameState,
    ) {
        let width = frame_state.window_cfg.width;
        let height = frame_state.window_cfg.height;

        dynamic_constants.push(FrameConstants {
            view_constants: ViewConstants::builder(frame_state.camera_matrices, width, height)
                .build(),
            mouse: gen_shader_mouse_state(&frame_state),
            frame_idx: self.frame_idx,
        });
    }

    fn retire_render_graph(&mut self, _retired_rg: &RetiredRenderGraph) {
        /*if let Some(handle) = self.sdf_img.last_rg_handle.take() {
            self.sdf_img.access_type = retired_rg.get_image(handle).1;
        }*/

        self.frame_idx = self.frame_idx.overflowing_add(1).0;
    }
}

/*// Vertices: bits 0, 1, 2, map to +/- X, Y, Z
fn cube_indices() -> Vec<u32> {
    let mut res = Vec::with_capacity(6 * 2 * 3);

    for (ndim, dim0, dim1) in [(1, 2, 4), (2, 4, 1), (4, 1, 2)].iter().copied() {
        for (nbit, dim0, dim1) in [(0, dim1, dim0), (ndim, dim0, dim1)].iter().copied() {
            res.push(nbit);
            res.push(nbit + dim0);
            res.push(nbit + dim1);

            res.push(nbit + dim1);
            res.push(nbit + dim0);
            res.push(nbit + dim0 + dim1);
        }
    }

    res
}*/

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

#[allow(dead_code)]
struct TemporalImage {
    resource: Arc<Image>,
    access_type: vk_sync::AccessType,
    last_rg_handle: Option<rg::ExportedHandle<Image>>,
}

#[allow(dead_code)]
impl TemporalImage {
    pub fn new(resource: Arc<Image>) -> Self {
        Self {
            resource,
            access_type: vk_sync::AccessType::Nothing,
            last_rg_handle: None,
        }
    }
}
