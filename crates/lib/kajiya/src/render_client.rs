use crate::{
    camera::CameraMatrices,
    frame_state::FrameState,
    image_lut::{ComputeImageLut, ImageLut},
    renderers::{
        csgi::CsgiRenderer, raster_meshes::*, rtdgi::RtdgiRenderer, rtr::*, ssgi::*,
        taa::TaaRenderer,
    },
    viewport::ViewConstants,
};
use glam::{Vec2, Vec3};
use kajiya_asset::mesh::{AssetRef, GpuImage, PackedTriMesh, PackedVertex};
use kajiya_backend::{
    ash::{
        version::DeviceV1_0,
        vk::{self, ImageView},
    },
    dynamic_constants::DynamicConstants,
    rspirv_reflect,
    vk_sync::{self, AccessType},
    vulkan::{self, image::*, shader::*, RenderBackend},
    vulkan::{
        device,
        ray_tracing::{
            RayTracingAcceleration, RayTracingBottomAccelerationDesc, RayTracingGeometryDesc,
            RayTracingGeometryPart, RayTracingGeometryType, RayTracingInstanceDesc,
            RayTracingTopAccelerationDesc,
        },
    },
};
use kajiya_rg::{self as rg, renderer::*, RetiredRenderGraph};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use parking_lot::Mutex;
use std::{collections::HashMap, mem::size_of, ops::Range, sync::Arc};
use vulkan::buffer::{Buffer, BufferDesc};

#[repr(C)]
#[derive(Copy, Clone)]
struct FrameConstants {
    view_constants: ViewConstants,
    sun_direction: [f32; 4],
    frame_idx: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct GpuMesh {
    vertex_core_offset: u32,
    vertex_uv_offset: u32,
    vertex_mat_offset: u32,
    vertex_aux_offset: u32,
    vertex_tangent_offset: u32,

    mat_data_offset: u32,
    index_offset: u32,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct MeshHandle(pub usize);

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct InstanceHandle(pub usize);

const MAX_GPU_MESHES: usize = 1024;
const VERTEX_BUFFER_CAPACITY: usize = 1024 * 1024 * 512;

#[derive(Clone, Copy)]
pub struct MeshInstance {
    pub position: Vec3,
    pub prev_position: Vec3,
    pub mesh: MeshHandle,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RenderDebugMode {
    None,
    CsgiVoxelGrid,
}

pub struct KajiyaRenderClient {
    device: Arc<device::Device>,

    pub(super) raster_simple_render_pass: Arc<RenderPass>,
    pub(super) bindless_descriptor_set: vk::DescriptorSet,
    pub(super) meshes: Vec<UploadedTriMesh>,
    pub(super) instances: Vec<MeshInstance>,
    pub(super) vertex_buffer: Mutex<Arc<Buffer>>,
    vertex_buffer_written: u64,

    mesh_buffer: Mutex<Arc<Buffer>>,

    mesh_blas: Vec<Arc<RayTracingAcceleration>>,
    tlas: Option<Arc<RayTracingAcceleration>>,

    bindless_images: Vec<Arc<Image>>,
    next_bindless_image_id: usize,

    image_luts: Vec<ImageLut>,
    frame_idx: u32,
    prev_camera_matrices: Option<CameraMatrices>,

    supersample_offsets: Vec<Vec2>,

    pub render_mode: RenderMode,
    pub reset_reference_accumulation: bool,

    pub ssgi: SsgiRenderer,
    pub rtr: RtrRenderer,
    pub rtdgi: RtdgiRenderer,
    pub csgi: CsgiRenderer,
    pub taa: TaaRenderer,

    pub debug_mode: RenderDebugMode,
    pub debug_shading_mode: usize,
    pub ev_shift: f32,

    pub ui_frame: Option<(UiRenderCallback, Arc<Image>)>,
}

pub type UiRenderCallback = Box<dyn FnOnce(vk::CommandBuffer) + 'static>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Standard,
    Reference,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BindlessImageHandle(pub u32);

/*fn as_byte_slice_unchecked<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * size_of::<T>()) }
}*/

/*fn append_buffer_data<T: Copy>(buf_slice: &mut [u8], buf_written: &mut usize, data: &[T]) -> usize {
    if !data.is_empty() {
        let alignment = std::mem::align_of::<T>();
        assert!(alignment.count_ones() == 1);

        let data_start = (*buf_written + alignment - 1) & !(alignment - 1);
        let data_bytes = data.len() * size_of::<T>();
        assert!(
            data_start + data_bytes <= buf_slice.len(),
            format!("data_bytes: {}; buf_slice: {}", data_bytes, buf_slice.len())
        );

        let dst = unsafe {
            std::slice::from_raw_parts_mut(buf_slice.as_ptr().add(data_start) as *mut T, data.len())
        };
        dst.copy_from_slice(data);

        *buf_written = data_start + data_bytes;
        data_start
    } else {
        0
    }
}*/

fn create_bindless_descriptor_set(device: &device::Device) -> vk::DescriptorSet {
    let raw_device = &device.raw;

    let set_binding_flags = [
        vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
    ];

    let mut binding_flags_create_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
        .binding_flags(&set_binding_flags)
        .build();

    let descriptor_set_layout = unsafe {
        raw_device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::ALL)
                            .build(),
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::ALL)
                            .build(),
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(2)
                            .descriptor_count(MAX_BINDLESS_DESCRIPTOR_COUNT as _)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .stage_flags(vk::ShaderStageFlags::ALL)
                            .build(),
                    ])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags_create_info)
                    .build(),
                None,
            )
            .unwrap()
    };

    let descriptor_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 2,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: MAX_BINDLESS_DESCRIPTOR_COUNT as _,
        },
    ];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&descriptor_sizes)
        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
        .max_sets(1);

    let descriptor_pool = unsafe {
        raw_device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap()
    };

    let variable_descriptor_count = MAX_BINDLESS_DESCRIPTOR_COUNT as _;
    let mut variable_descriptor_count_allocate_info =
        vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
            .descriptor_counts(std::slice::from_ref(&variable_descriptor_count))
            .build();

    let set = unsafe {
        raw_device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                    .push_next(&mut variable_descriptor_count_allocate_info)
                    .build(),
            )
            .unwrap()[0]
    };

    set
}

trait BufferDataSource {
    fn as_bytes(&self) -> &[u8];
    fn alignment(&self) -> u64;
}

struct PendingBufferUpload {
    source: Box<dyn BufferDataSource>,
    offset: u64,
}

impl<T: Copy> BufferDataSource for &'static [T] {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const u8,
                self.len() * std::mem::size_of::<T>(),
            )
        }
    }

    fn alignment(&self) -> u64 {
        std::mem::align_of::<T>() as u64
    }
}

impl<T: Copy> BufferDataSource for Vec<T> {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const u8,
                self.len() * std::mem::size_of::<T>(),
            )
        }
    }

    fn alignment(&self) -> u64 {
        std::mem::align_of::<T>() as u64
    }
}
struct BufferBuilder {
    //buf_slice: &'a mut [u8],
    pending_uploads: Vec<PendingBufferUpload>,
    current_offset: u64,
}

impl Default for BufferBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferBuilder {
    fn new() -> Self {
        Self {
            pending_uploads: Vec::new(),
            current_offset: 0,
        }
    }

    fn append(&mut self, data: impl BufferDataSource + 'static) -> u64 {
        let alignment = data.alignment();
        assert!(alignment.count_ones() == 1);

        let data_start = (self.current_offset + alignment - 1) & !(alignment - 1);
        let data_len = data.as_bytes().len() as u64;

        self.pending_uploads.push(PendingBufferUpload {
            source: Box::new(data),
            offset: data_start,
        });
        self.current_offset = data_start + data_len;

        data_start
    }

    fn upload(self, device: &kajiya_backend::Device, target: &mut Buffer, target_offset: u64) {
        let target = target.raw;

        // TODO: share a common staging buffer, don't leak
        const STAGING_BYTES: usize = 64 * 1024 * 1024;
        let mut staging_buffer = device
            .create_buffer(
                BufferDesc {
                    size: STAGING_BYTES,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    mapped: true,
                },
                None,
            )
            .expect("Staging buffer for buffer upload");

        struct UploadChunk {
            pending_idx: usize,
            src_range: Range<usize>,
        }

        // TODO: merge chunks to perform fewer uploads if multiple source regions fit in one chunk
        let chunks: Vec<UploadChunk> = self
            .pending_uploads
            .iter()
            .enumerate()
            .flat_map(|(pending_idx, pending)| {
                let byte_count = pending.source.as_bytes().len();
                let chunk_count = (byte_count + STAGING_BYTES - 1) / STAGING_BYTES;
                (0..chunk_count).map(move |chunk| UploadChunk {
                    pending_idx,
                    src_range: chunk * STAGING_BYTES..((chunk + 1) * STAGING_BYTES).min(byte_count),
                })
            })
            .collect();

        for UploadChunk {
            pending_idx,
            src_range,
        } in chunks
        {
            let pending = &self.pending_uploads[pending_idx];
            staging_buffer.allocation.mapped_slice_mut().unwrap()
                [0..(src_range.end - src_range.start)]
                .copy_from_slice(&pending.source.as_bytes()[src_range.start..src_range.end]);

            device.with_setup_cb(|cb| unsafe {
                device.raw.cmd_copy_buffer(
                    cb,
                    staging_buffer.raw,
                    target,
                    &[vk::BufferCopy::builder()
                        .src_offset(0u64)
                        .dst_offset(target_offset + pending.offset + src_range.start as u64)
                        .size((src_range.end - src_range.start) as u64)
                        .build()],
                );
            });
        }
    }
}

fn load_gpu_image_asset(
    device: Arc<kajiya_backend::Device>,
    asset: AssetRef<GpuImage::Flat>,
) -> Arc<Image> {
    let asset = crate::mmap::mmapped_asset::<GpuImage::Flat>(&format!(
        "baked/{:8.8x}.image",
        asset.identity()
    ))
    .unwrap();

    let desc = ImageDesc::new_2d(asset.format, [asset.extent[0], asset.extent[1]])
        .usage(vk::ImageUsageFlags::SAMPLED)
        .mip_levels(asset.mips.len() as _);

    let initial_data = asset
        .mips
        .iter()
        .enumerate()
        .map(|(mip_level, mip)| ImageSubResourceData {
            data: mip.as_slice(),
            row_pitch: ((desc.extent[0] as usize) >> mip_level).max(1) * 4,
            slice_pitch: 0,
        })
        .collect::<Vec<_>>();

    Arc::new(device.create_image(desc, initial_data).unwrap())
}

impl KajiyaRenderClient {
    pub fn new(backend: &RenderBackend) -> anyhow::Result<Self> {
        let raster_simple_render_pass = create_render_pass(
            &*backend.device,
            RenderPassDesc {
                color_attachments: &[
                    // view-space geometry normal; * 2 - 1 to decode
                    RenderPassAttachmentDesc::new(vk::Format::A2R10G10B10_UNORM_PACK32)
                        .garbage_input(),
                    // gbuffer
                    RenderPassAttachmentDesc::new(vk::Format::R32G32B32A32_SFLOAT).garbage_input(),
                    // velocity
                    RenderPassAttachmentDesc::new(vk::Format::R16G16B16A16_SFLOAT).garbage_input(),
                ],
                depth_attachment: Some(RenderPassAttachmentDesc::new(
                    vk::Format::D24_UNORM_S8_UINT,
                )),
            },
        )?;

        let mesh_buffer = backend
            .device
            .create_buffer(
                BufferDesc {
                    size: MAX_GPU_MESHES * size_of::<GpuMesh>(),
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                    mapped: true,
                },
                None,
            )
            .unwrap();

        let vertex_buffer = backend
            .device
            .create_buffer(
                BufferDesc {
                    size: VERTEX_BUFFER_CAPACITY,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    mapped: false,
                },
                None,
            )
            .unwrap();

        let bindless_descriptor_set = create_bindless_descriptor_set(backend.device.as_ref());

        Self::write_descriptor_set_buffer(
            &backend.device.raw,
            bindless_descriptor_set,
            0,
            &mesh_buffer,
        );

        Self::write_descriptor_set_buffer(
            &backend.device.raw,
            bindless_descriptor_set,
            1,
            &vertex_buffer,
        );

        let supersample_offsets = (0..16)
            .map(|i| {
                Vec2::new(
                    radical_inverse(i % 16 + 1, 2) - 0.5,
                    radical_inverse(i % 16 + 1, 3) - 0.5,
                )
            })
            .collect();
        //let supersample_offsets = vec![Vec2::new(0.0, -0.5), Vec2::new(0.0, 0.5)];

        Ok(Self {
            raster_simple_render_pass,

            reset_reference_accumulation: false,
            //cube_index_buffer: Arc::new(cube_index_buffer),
            device: backend.device.clone(),
            meshes: Default::default(),
            mesh_blas: Default::default(),
            instances: Default::default(),
            tlas: Default::default(),
            mesh_buffer: Mutex::new(Arc::new(mesh_buffer)),
            vertex_buffer: Mutex::new(Arc::new(vertex_buffer)),
            vertex_buffer_written: 0,
            bindless_descriptor_set,
            bindless_images: Default::default(),
            image_luts: Default::default(),
            next_bindless_image_id: 0,
            render_mode: RenderMode::Standard,
            frame_idx: 0u32,
            prev_camera_matrices: None,

            supersample_offsets,

            ssgi: Default::default(),
            rtr: RtrRenderer::new(backend.device.as_ref()),
            csgi: CsgiRenderer::default(),
            rtdgi: RtdgiRenderer::new(backend.device.as_ref()),
            taa: TaaRenderer::new(),

            debug_mode: RenderDebugMode::None,
            debug_shading_mode: 0,
            ev_shift: 0.0,

            ui_frame: Default::default(),
        })
    }

    fn write_descriptor_set_buffer(
        device: &kajiya_backend::ash::Device,
        set: vk::DescriptorSet,
        dst_binding: u32,
        buffer: &Buffer,
    ) {
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(buffer.raw)
            .range(vk::WHOLE_SIZE)
            .build();

        let write_descriptor_set = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_binding(dst_binding)
            .buffer_info(std::slice::from_ref(&buffer_info))
            .build();

        unsafe {
            device.update_descriptor_sets(std::slice::from_ref(&write_descriptor_set), &[]);
        }
    }

    fn add_bindless_image_view(&mut self, view: ImageView) -> BindlessImageHandle {
        let handle = BindlessImageHandle(self.next_bindless_image_id as _);
        self.next_bindless_image_id += 1;

        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .build();

        let write_descriptor_set = vk::WriteDescriptorSet::builder()
            .dst_set(self.bindless_descriptor_set)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_binding(2)
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

    pub fn add_image_lut(&mut self, computer: impl ComputeImageLut + 'static, id: usize) {
        self.image_luts
            .push(ImageLut::new(self.device.as_ref(), Box::new(computer)));

        let handle = self.add_bindless_image_view(
            self.image_luts
                .last()
                .unwrap()
                .backing_image()
                .view(self.device.as_ref(), &ImageViewDesc::default()),
        );

        assert_eq!(handle.0 as usize, id);
    }

    pub fn add_image(&mut self, image: Arc<Image>) -> BindlessImageHandle {
        let handle = self
            .add_bindless_image_view(image.view(self.device.as_ref(), &ImageViewDesc::default()));
        self.bindless_images.push(image);
        handle
    }

    pub fn add_mesh(&mut self, mesh: &'static PackedTriMesh::Flat) -> MeshHandle {
        let mesh_idx = self.meshes.len();
        let mut unique_images: Vec<AssetRef<GpuImage::Flat>> = mesh.maps.as_slice().to_vec();
        unique_images.sort();
        unique_images.dedup();

        let loaded_images = {
            let device = self.device.clone();
            easy_parallel::Parallel::new()
                .each(unique_images.iter(), |&asset| {
                    load_gpu_image_asset(device, asset)
                })
                .run()
        };
        /*let loaded_images = {
            let device = self.device.clone();
            unique_images
                .iter()
                .map(|&asset| load_gpu_image_asset(device.clone(), asset))
                .collect::<Vec<_>>()
        };*/
        let loaded_images = loaded_images.into_iter().map(|img| self.add_image(img));

        let material_map_to_image: HashMap<AssetRef<GpuImage::Flat>, BindlessImageHandle> =
            unique_images.into_iter().zip(loaded_images).collect();

        let mut materials = mesh.materials.as_slice().to_vec();
        {
            let mesh_map_gpu_ids: Vec<BindlessImageHandle> = mesh
                .maps
                .as_slice()
                .iter()
                .map(|map| material_map_to_image[map])
                .collect();

            for mat in &mut materials {
                for m in &mut mat.maps {
                    *m = mesh_map_gpu_ids[*m as usize].0;
                }
            }
        }

        let vertex_data_offset = self.vertex_buffer_written as u32;

        let mut buffer_builder = BufferBuilder::new();
        let vertex_index_offset =
            buffer_builder.append(mesh.indices.as_slice()) as u32 + vertex_data_offset;
        let vertex_core_offset =
            buffer_builder.append(mesh.verts.as_slice()) as u32 + vertex_data_offset;
        let vertex_uv_offset =
            buffer_builder.append(mesh.uvs.as_slice()) as u32 + vertex_data_offset;
        let vertex_mat_offset =
            buffer_builder.append(mesh.material_ids.as_slice()) as u32 + vertex_data_offset;
        let vertex_aux_offset =
            buffer_builder.append(mesh.colors.as_slice()) as u32 + vertex_data_offset;
        let vertex_tangent_offset =
            buffer_builder.append(mesh.tangents.as_slice()) as u32 + vertex_data_offset;
        let mat_data_offset = buffer_builder.append(materials) as u32 + vertex_data_offset;

        let total_buffer_size = buffer_builder.current_offset;
        let mut vertex_buffer = self.vertex_buffer.lock();
        buffer_builder.upload(
            self.device.as_ref(),
            Arc::get_mut(&mut *vertex_buffer).expect("refs may not be retained"),
            self.vertex_buffer_written,
        );
        self.vertex_buffer_written += total_buffer_size;

        let mesh_buffer_dst = unsafe {
            let mut mesh_buffer = self.mesh_buffer.lock();
            let mesh_buffer = Arc::get_mut(&mut *mesh_buffer).expect("refs may not be retained");
            let mesh_buffer_dst =
                mesh_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut GpuMesh;
            std::slice::from_raw_parts_mut(mesh_buffer_dst, MAX_GPU_MESHES)
        };

        let base_da = vertex_buffer.device_address(&self.device);
        let vertex_buffer_da = base_da + vertex_core_offset as u64;
        let index_buffer_da = base_da + vertex_index_offset as u64;

        let blas = self
            .device
            .create_ray_tracing_bottom_acceleration(&RayTracingBottomAccelerationDesc {
                geometries: vec![RayTracingGeometryDesc {
                    geometry_type: RayTracingGeometryType::Triangle,
                    vertex_buffer: vertex_buffer_da,
                    index_buffer: index_buffer_da,
                    vertex_format: vk::Format::R32G32B32_SFLOAT,
                    vertex_stride: size_of::<PackedVertex>(),
                    parts: vec![RayTracingGeometryPart {
                        index_count: mesh.indices.len(),
                        index_offset: 0,
                        max_vertex: mesh
                            .indices
                            .as_slice()
                            .iter()
                            .copied()
                            .max()
                            .expect("mesh must not be empty"),
                    }],
                }],
            })
            .expect("blas");

        mesh_buffer_dst[mesh_idx] = GpuMesh {
            vertex_core_offset,
            vertex_uv_offset,
            vertex_mat_offset,
            vertex_aux_offset,
            vertex_tangent_offset,
            mat_data_offset,
            index_offset: vertex_index_offset,
        };

        self.meshes.push(UploadedTriMesh {
            index_buffer_offset: vertex_index_offset as u64,
            index_count: mesh.indices.len() as _,
        });

        self.mesh_blas.push(Arc::new(blas));

        MeshHandle(mesh_idx)
    }

    pub fn add_instance(&mut self, mesh: MeshHandle, pos: Vec3) -> InstanceHandle {
        let handle = InstanceHandle(self.instances.len());
        self.instances.push(MeshInstance {
            position: pos,
            prev_position: pos,
            mesh,
        });
        handle
    }

    #[allow(dead_code)]
    pub fn set_instance_transform(&mut self, inst: InstanceHandle, pos: Vec3) {
        self.instances[inst.0].position = pos;
    }

    pub fn build_ray_tracing_top_level_acceleration(&mut self) {
        let tlas = self
            .device
            .create_ray_tracing_top_acceleration(&RayTracingTopAccelerationDesc {
                //instances: self.mesh_blas.iter().collect::<Vec<_>>(),
                instances: self
                    .instances
                    .iter()
                    .map(|inst| RayTracingInstanceDesc {
                        blas: self.mesh_blas[inst.mesh.0].clone(),
                        position: inst.position,
                        mesh_index: inst.mesh.0 as u32,
                    })
                    .collect::<Vec<_>>(),
            })
            .expect("tlas");

        self.tlas = Some(Arc::new(tlas));
    }

    #[allow(dead_code)]
    pub fn reset_frame_idx(&mut self) {
        self.frame_idx = 0;
    }
}

impl KajiyaRenderClient {
    pub(super) fn prepare_top_level_acceleration(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
    ) -> rg::Handle<RayTracingAcceleration> {
        let mut tlas = rg.import(
            self.tlas.as_ref().unwrap().clone(),
            vk_sync::AccessType::AnyShaderReadOther,
        );

        let instances = self
            .instances
            .iter()
            .map(|inst| RayTracingInstanceDesc {
                blas: self.mesh_blas[inst.mesh.0].clone(),
                position: inst.position,
                mesh_index: inst.mesh.0 as u32,
            })
            .collect::<Vec<_>>();

        let mut pass = rg.add_pass("rebuild tlas");
        let tlas_ref = pass.write(&mut tlas, AccessType::TransferWrite);

        pass.render(move |api| {
            //let device = &api.device().raw;
            let resources = &mut api.resources;
            let instance_buffer_address = resources
                .execution_params
                .device
                .fill_ray_tracing_instance_buffer(resources.dynamic_constants, &instances);
            let tlas = api.resources.rt_acceleration(tlas_ref);

            let cb = api.cb;
            api.device().rebuild_ray_tracing_top_acceleration(
                cb.raw,
                instance_buffer_address,
                instances.len(),
                tlas,
            );
        });

        tlas
    }

    pub fn store_prev_mesh_transforms(&mut self) {
        for inst in &mut self.instances {
            inst.prev_position = inst.position;
        }
    }
}

lazy_static::lazy_static! {
    static ref BINDLESS_DESCRIPTOR_SET_LAYOUT: HashMap<u32, rspirv_reflect::DescriptorInfo> = [
        (0, rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::STORAGE_BUFFER,
            is_bindless: false,
            name: Default::default(),
        }),
        (1, rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::STORAGE_BUFFER,
            is_bindless: false,
            name: Default::default(),
        }),
        (2, rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::SAMPLED_IMAGE,
            is_bindless: true,
            name: Default::default(),
        }),
    ]
    .iter()
    .cloned()
    .collect();
}

impl RenderClient<FrameState> for KajiyaRenderClient {
    fn prepare_render_graph(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        frame_state: &FrameState,
    ) -> RenderGraphOutput {
        rg.predefined_descriptor_set_layouts.insert(
            1,
            rg::PredefinedDescriptorSet {
                bindings: BINDLESS_DESCRIPTOR_SET_LAYOUT.clone(),
            },
        );

        for image_lut in self.image_luts.iter_mut() {
            image_lut.compute_if_needed(rg);
        }

        let main_img = match self.render_mode {
            RenderMode::Standard => self.prepare_render_graph_standard(rg, frame_state),
            RenderMode::Reference => self.prepare_render_graph_reference(rg, frame_state),
        };

        let ui_img = self.render_ui(rg);
        let ui_img = rg.export(
            ui_img,
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        RenderGraphOutput { main_img, ui_img }
    }

    fn prepare_frame_constants(
        &mut self,
        dynamic_constants: &mut DynamicConstants,
        frame_state: &FrameState,
    ) {
        let width = frame_state.window_cfg.width;
        let height = frame_state.window_cfg.height;

        let mut view_constants = ViewConstants::builder(
            frame_state.camera_matrices,
            self.prev_camera_matrices
                .unwrap_or(frame_state.camera_matrices),
            width,
            height,
        )
        .build();

        // Re-shuffle the jitter sequence if we've just used it up
        /*if 0 == self.frame_idx % self.samples.len() as u32 && self.frame_idx > 0 {
            use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
            let mut rng = SmallRng::seed_from_u64(self.frame_idx as u64);

            let prev_sample = self.samples.last().copied();
            loop {
                // Will most likely shuffle only once. Re-shuffles if the first sample
                // in the new sequence is the same as the last sample in the last.
                self.samples.shuffle(&mut rng);
                if self.samples.first().copied() != prev_sample {
                    break;
                }
            }
        }*/

        if self.render_mode == RenderMode::Standard {
            let supersample_offset =
                self.supersample_offsets[self.frame_idx as usize % self.supersample_offsets.len()];
            //Vec2::zero();
            self.taa.current_supersample_offset = supersample_offset;
            view_constants.set_pixel_offset(supersample_offset, width, height);
        }

        dynamic_constants.push(&FrameConstants {
            view_constants,
            sun_direction: [
                frame_state.sun_direction.x,
                frame_state.sun_direction.y,
                frame_state.sun_direction.z,
                0.0,
            ],
            frame_idx: self.frame_idx,
        });

        self.prev_camera_matrices = Some(frame_state.camera_matrices);
    }

    fn retire_render_graph(&mut self, _rg: &RetiredRenderGraph) {
        self.frame_idx = self.frame_idx.overflowing_add(1).0;
    }
}

fn radical_inverse(mut n: u32, base: u32) -> f32 {
    let mut val = 0.0f32;
    let inv_base = 1.0f32 / base as f32;
    let mut inv_bi = inv_base;

    while n > 0 {
        let d_i = n % base;
        val += d_i as f32 * inv_bi;
        n = (n as f32 * inv_base) as u32;
        inv_bi *= inv_base;
    }

    val
}
