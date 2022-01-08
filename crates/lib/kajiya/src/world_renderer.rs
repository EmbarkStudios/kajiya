use crate::{
    bindless_descriptor_set::{create_bindless_descriptor_set, BINDLESS_DESCRIPTOR_SET_LAYOUT},
    buffer_builder::BufferBuilder,
    frame_desc::WorldFrameDesc,
    image_lut::{ComputeImageLut, ImageLut},
    renderers::{
        csgi::CsgiRenderer, lighting::LightingRenderer, raster_meshes::*, rtdgi::RtdgiRenderer,
        rtr::*, shadow_denoise::ShadowDenoiseRenderer, ssgi::*, taa::TaaRenderer,
    },
};
use glam::{Affine3A, Vec2, Vec3};
use kajiya_asset::mesh::{AssetRef, GpuImage, MeshMaterialFlags, PackedTriMesh, PackedVertex};
use kajiya_backend::{
    ash::vk::{self, ImageView},
    dynamic_constants::DynamicConstants,
    vk_sync::{self, AccessType},
    vulkan::{self, device, image::*, ray_tracing::*, shader::*, RenderBackend},
};
use kajiya_rg::{self as rg};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use parking_lot::Mutex;
use rg::renderer::FrameConstantsLayout;
use rust_shaders_shared::{
    camera::CameraMatrices,
    frame_constants::{FrameConstants, GiCascadeConstants, MAX_CSGI_CASCADE_COUNT},
    view_constants::ViewConstants,
};
use std::{collections::HashMap, mem::size_of, sync::Arc};
use vulkan::buffer::{Buffer, BufferDesc};

#[cfg(feature = "dlss")]
use crate::renderers::dlss::DlssRenderer;

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
const TLAS_PREALLOCATE_BYTES: usize = 1024 * 1024 * 32;

#[derive(Clone, Copy)]
pub struct InstanceDynamicParameters {
    pub emissive_multiplier: f32,
}

impl Default for InstanceDynamicParameters {
    fn default() -> Self {
        Self {
            emissive_multiplier: 1.0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct MeshInstance {
    pub transformation: Affine3A,
    pub prev_transformation: Affine3A,
    pub mesh: MeshHandle,
    pub dynamic_parameters: InstanceDynamicParameters,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RenderDebugMode {
    None,
    CsgiVoxelGrid { cascade_idx: usize },
    CsgiRadiance,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct TriangleLight {
    pub verts: [[f32; 3]; 3],
    pub radiance: [f32; 3],
}

impl TriangleLight {
    pub fn transform(
        self,
        translation: Vec3,
        rotation: impl core::ops::Mul<Vec3, Output = Vec3> + Copy,
    ) -> Self {
        Self {
            verts: [
                (rotation * Vec3::from(self.verts[0]) + translation).into(),
                (rotation * Vec3::from(self.verts[1]) + translation).into(),
                (rotation * Vec3::from(self.verts[2]) + translation).into(),
            ],
            radiance: self.radiance,
        }
    }

    pub fn scale_radiance(self, scale: Vec3) -> Self {
        Self {
            verts: self.verts,
            radiance: (Vec3::from(self.radiance) * scale).into(),
        }
    }
}

pub struct MeshLightSet {
    pub lights: Vec<TriangleLight>,
}

pub struct WorldRenderer {
    device: Arc<device::Device>,

    pub(super) raster_simple_render_pass: Arc<RenderPass>,
    pub(super) bindless_descriptor_set: vk::DescriptorSet,
    pub(super) meshes: Vec<UploadedTriMesh>,

    pub(super) mesh_lights: Vec<MeshLightSet>,

    // ----
    // SoA
    pub(super) instances: Vec<MeshInstance>,
    pub(super) instance_handles: Vec<InstanceHandle>,
    // ----

    // The `usize` indexes into `instances` and `instance_handles`
    pub(super) instance_handle_to_index: HashMap<InstanceHandle, usize>,

    pub(super) vertex_buffer: Mutex<Arc<Buffer>>,
    vertex_buffer_written: u64,

    mesh_buffer: Mutex<Arc<Buffer>>,

    mesh_blas: Vec<Arc<RayTracingAcceleration>>,
    tlas: Option<Arc<RayTracingAcceleration>>,
    accel_scratch: RayTracingAccelerationScratchBuffer,

    bindless_images: Vec<Arc<Image>>,
    next_bindless_image_id: usize,
    next_instance_handle: usize,

    image_luts: Vec<ImageLut>,
    frame_idx: u32,
    prev_camera_matrices: Option<CameraMatrices>,
    pub(crate) temporal_upscale_extent: [u32; 2],

    supersample_offsets: Vec<Vec2>,

    pub rg_debug_hook: Option<rg::GraphDebugHook>,
    pub render_mode: RenderMode,
    pub reset_reference_accumulation: bool,

    pub ssgi: SsgiRenderer,
    pub rtr: RtrRenderer,
    pub lighting: LightingRenderer,
    pub rtdgi: RtdgiRenderer,
    pub csgi: CsgiRenderer,
    pub taa: TaaRenderer,
    pub shadow_denoise: ShadowDenoiseRenderer,

    #[cfg(feature = "dlss")]
    pub dlss: DlssRenderer,
    #[cfg(feature = "dlss")]
    pub use_dlss: bool,

    pub debug_mode: RenderDebugMode,
    pub debug_shading_mode: usize,
    pub ev_shift: f32,

    pub world_gi_scale: f32,
    pub sun_size_multiplier: f32,
    pub sun_color_multiplier: Vec3,
    pub sky_ambient: Vec3,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Standard,
    Reference,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BindlessImageHandle(pub u32);

fn load_gpu_image_asset(
    device: Arc<kajiya_backend::Device>,
    asset: AssetRef<GpuImage::Flat>,
) -> Arc<Image> {
    let asset = crate::mmap::mmapped_asset::<GpuImage::Flat, _>(&format!(
        "/baked/{:8.8x}.image",
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

#[derive(Default)]
pub struct AddMeshOptions {
    pub use_lights: bool,
}

impl AddMeshOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn use_lights(mut self, v: bool) -> Self {
        self.use_lights = v;
        self
    }
}

impl WorldRenderer {
    pub(crate) fn new_empty(
        // Internal render resolution, before any upsampling
        #[allow(unused_variables)] render_extent: [u32; 2],
        temporal_upscale_extent: [u32; 2],
        backend: &RenderBackend,
    ) -> anyhow::Result<Self> {
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
                depth_attachment: Some(RenderPassAttachmentDesc::new(vk::Format::D32_SFLOAT)),
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
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
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

        let supersample_count = 128;
        let supersample_offsets = (1..=supersample_count)
            .map(|i| Vec2::new(radical_inverse(i, 2) - 0.5, radical_inverse(i, 3) - 0.5))
            .collect();
        //let supersample_offsets = vec![Vec2::new(0.0, -0.5), Vec2::new(0.0, 0.5)];
        /*let supersample_offsets = vec![
            Vec2::new(0.25, 0.25),
            Vec2::new(0.25, -0.25),
            Vec2::new(-0.25, 0.25),
            Vec2::new(-0.25, -0.25),
        ];*/

        let accel_scratch = backend
            .device
            .create_ray_tracing_acceleration_scratch_buffer()?;

        #[cfg(feature = "dlss")]
        let dlss = DlssRenderer::new(backend, render_extent, temporal_upscale_extent);

        Ok(Self {
            raster_simple_render_pass,

            reset_reference_accumulation: false,
            //cube_index_buffer: Arc::new(cube_index_buffer),
            device: backend.device.clone(),
            meshes: Default::default(),
            instances: Default::default(),
            instance_handles: Default::default(),
            instance_handle_to_index: Default::default(),

            mesh_lights: Default::default(),

            mesh_blas: Default::default(),
            tlas: Default::default(),
            accel_scratch,

            mesh_buffer: Mutex::new(Arc::new(mesh_buffer)),
            vertex_buffer: Mutex::new(Arc::new(vertex_buffer)),
            vertex_buffer_written: 0,
            bindless_descriptor_set,
            bindless_images: Default::default(),
            image_luts: Default::default(),

            next_bindless_image_id: 0,
            next_instance_handle: 0,

            rg_debug_hook: None,
            render_mode: RenderMode::Standard,
            frame_idx: 0u32,
            prev_camera_matrices: None,

            supersample_offsets,

            ssgi: Default::default(),
            rtr: RtrRenderer::new(backend.device.as_ref()),
            lighting: LightingRenderer::new(),
            csgi: CsgiRenderer::default(),
            rtdgi: RtdgiRenderer::new(backend.device.as_ref()),
            taa: TaaRenderer::new(),
            shadow_denoise: Default::default(),

            #[cfg(feature = "dlss")]
            dlss,
            #[cfg(feature = "dlss")]
            use_dlss: true,

            temporal_upscale_extent,

            debug_mode: RenderDebugMode::None,
            debug_shading_mode: 0,
            ev_shift: 0.0,
            world_gi_scale: 1.0,
            sun_size_multiplier: 1.0, // Sun as seen from Earth
            sun_color_multiplier: Vec3::ONE,
            sky_ambient: Vec3::ZERO,
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

    pub fn add_mesh(
        &mut self,
        mesh: &'static PackedTriMesh::Flat,
        opts: AddMeshOptions,
    ) -> MeshHandle {
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

        // If using emissives as lights, flag it in the material parameters
        if opts.use_lights {
            for mat in materials.iter_mut() {
                mat.flags |= MeshMaterialFlags::MESH_MATERIAL_FLAG_EMISSIVE_USED_AS_LIGHT;
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

        let total_buffer_size = buffer_builder.current_offset();
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
            .create_ray_tracing_bottom_acceleration(
                &RayTracingBottomAccelerationDesc {
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
                },
                &self.accel_scratch,
            )
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

        let mesh_lights = if opts.use_lights {
            let emissive_materials = mesh
                .materials
                .iter()
                .map(|mat| mat.emissive[0] > 0.0 || mat.emissive[1] > 0.0 || mat.emissive[2] > 0.0)
                .collect::<Vec<bool>>();

            let mut mesh_lights: Vec<TriangleLight> = Vec::new();
            for indices in mesh.indices.as_slice().chunks_exact(3) {
                let mat_idx = mesh.material_ids[indices[0] as usize] as usize;
                if !emissive_materials[mat_idx] {
                    continue;
                }

                let v0 = mesh.verts[indices[0] as usize].pos;
                let v1 = mesh.verts[indices[1] as usize].pos;
                let v2 = mesh.verts[indices[2] as usize].pos;
                let radiance = mesh.materials[mat_idx].emissive;

                mesh_lights.push(TriangleLight {
                    verts: [v0, v1, v2],
                    radiance,
                });
            }

            mesh_lights
        } else {
            Vec::new()
        };

        self.mesh_lights.push(MeshLightSet {
            lights: mesh_lights,
        });

        MeshHandle(mesh_idx)
    }

    pub fn add_instance(&mut self, mesh: MeshHandle, transform: Affine3A) -> InstanceHandle {
        let handle = self.next_instance_handle;
        self.next_instance_handle += 1;
        let handle = InstanceHandle(handle);

        let index = self.instances.len();

        self.instances.push(MeshInstance {
            transformation: transform,
            prev_transformation: transform,
            mesh,
            dynamic_parameters: InstanceDynamicParameters::default(),
        });
        self.instance_handles.push(handle);

        assert_eq!(self.instances.len(), self.instance_handles.len());

        self.instance_handle_to_index.insert(handle, index);

        handle
    }

    pub fn remove_instance(&mut self, inst: InstanceHandle) {
        let index = self
            .instance_handle_to_index
            .remove(&inst)
            .expect("no such instance");
        self.instances.swap_remove(index);
        self.instance_handles.swap_remove(index);

        // A new instance could have been moved into this slot in the vec.
        // Make sure `instance_handle_to_index` reflects this.
        if let Some(new_handle) = self.instance_handles.get(index).copied() {
            self.instance_handle_to_index.insert(new_handle, index);
        }
    }

    pub fn set_instance_transform(&mut self, inst: InstanceHandle, transform: Affine3A) {
        let index = self.instance_handle_to_index[&inst];
        self.instances[index].transformation = transform;
    }

    pub fn get_instance_dynamic_parameters(
        &self,
        inst: InstanceHandle,
    ) -> &InstanceDynamicParameters {
        let index = self.instance_handle_to_index[&inst];
        &self.instances[index].dynamic_parameters
    }

    pub fn get_instance_dynamic_parameters_mut(
        &mut self,
        inst: InstanceHandle,
    ) -> &mut InstanceDynamicParameters {
        let index = self.instance_handle_to_index[&inst];
        &mut self.instances[index].dynamic_parameters
    }

    pub(crate) fn build_ray_tracing_top_level_acceleration(&mut self) {
        let tlas = self
            .device
            .create_ray_tracing_top_acceleration(
                &RayTracingTopAccelerationDesc {
                    //instances: self.mesh_blas.iter().collect::<Vec<_>>(),
                    instances: self
                        .instances
                        .iter()
                        .map(|inst| RayTracingInstanceDesc {
                            blas: self.mesh_blas[inst.mesh.0].clone(),
                            transformation: inst.transformation,
                            mesh_index: inst.mesh.0 as u32,
                        })
                        .collect::<Vec<_>>(),
                    preallocate_bytes: TLAS_PREALLOCATE_BYTES,
                },
                &self.accel_scratch,
            )
            .expect("tlas");

        self.tlas = Some(Arc::new(tlas));
    }

    #[allow(dead_code)]
    pub fn reset_frame_idx(&mut self) {
        self.frame_idx = 0;
    }

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
                transformation: inst.transformation,
                mesh_index: inst.mesh.0 as u32,
            })
            .collect::<Vec<_>>();

        let mut pass = rg.add_pass("rebuild tlas");
        let tlas_ref = pass.write(&mut tlas, AccessType::TransferWrite);

        let accel_scratch = self.accel_scratch.clone();

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
                &accel_scratch,
            );
        });

        tlas
    }

    fn store_prev_mesh_transforms(&mut self) {
        for inst in &mut self.instances {
            inst.prev_transformation = inst.transformation;
        }
    }

    pub fn prepare_render_graph(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        frame_desc: &WorldFrameDesc,
    ) -> rg::Handle<Image> {
        rg.predefined_descriptor_set_layouts.insert(
            1,
            rg::PredefinedDescriptorSet {
                bindings: BINDLESS_DESCRIPTOR_SET_LAYOUT.clone(),
            },
        );

        for image_lut in self.image_luts.iter_mut() {
            image_lut.compute_if_needed(rg);
        }

        match self.render_mode {
            RenderMode::Standard => {
                self.taa.current_supersample_offset = self.supersample_offsets
                    [self.frame_idx as usize % self.supersample_offsets.len()];
                //self.taa.current_supersample_offset = Vec2::ZERO;

                #[cfg(feature = "dlss")]
                {
                    self.dlss.current_supersample_offset = self.taa.current_supersample_offset;
                }

                self.prepare_render_graph_standard(rg, frame_desc)
            }
            RenderMode::Reference => {
                self.taa.current_supersample_offset = Vec2::ZERO;

                #[cfg(feature = "dlss")]
                {
                    self.dlss.current_supersample_offset = self.taa.current_supersample_offset;
                }

                self.prepare_render_graph_reference(rg, frame_desc)
            }
        }
    }

    pub fn prepare_frame_constants(
        &mut self,
        dynamic_constants: &mut DynamicConstants,
        frame_desc: &WorldFrameDesc,
        delta_time_seconds: f32,
    ) -> FrameConstantsLayout {
        let mut view_constants = ViewConstants::builder(
            frame_desc.camera_matrices,
            self.prev_camera_matrices
                .unwrap_or(frame_desc.camera_matrices),
            frame_desc.render_extent,
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

        view_constants.set_pixel_offset(
            self.taa.current_supersample_offset,
            frame_desc.render_extent.into(),
        );

        let triangle_lights: Vec<TriangleLight> = self
            .instances
            .iter()
            .flat_map(|inst| {
                let (_scale, rotation, translation) =
                    inst.transformation.to_scale_rotation_translation();
                let inst_position = translation;
                let inst_rotation = rotation;

                let emissive_multiplier = Vec3::splat(inst.dynamic_parameters.emissive_multiplier);

                self.mesh_lights[inst.mesh.0]
                    .lights
                    .iter()
                    .map(move |light: &TriangleLight| {
                        light
                            .transform(inst_position, inst_rotation)
                            .scale_radiance(emissive_multiplier)
                    })
            })
            .collect();

        // Initialize constants for the maximum allowed cascade count, even if we're not using them,
        // so that we don't need to change the layout of frame constants up to this limit.
        let mut gi_cascades: [GiCascadeConstants; MAX_CSGI_CASCADE_COUNT] = Default::default();

        self.csgi
            .update_eye_position(&view_constants.eye_position(), self.world_gi_scale);

        // Actually set the cascade constants we're using
        for (i, c) in self
            .csgi
            .constants(self.world_gi_scale)
            .iter()
            .copied()
            .enumerate()
        {
            gi_cascades[i] = c;
        }

        let real_sun_angular_radius = 0.53f32.to_radians() * 0.5;

        let globals_offset = dynamic_constants.push(&FrameConstants {
            view_constants,
            sun_direction: frame_desc.sun_direction.extend(0.0),
            frame_index: self.frame_idx,
            delta_time_seconds,
            sun_angular_radius_cos: (self.sun_size_multiplier * real_sun_angular_radius).cos(),

            sun_color_multiplier: self.sun_color_multiplier.extend(0.0),
            sky_ambient: self.sky_ambient.extend(0.0),
            triangle_light_count: triangle_lights.len() as _,
            world_gi_scale: self.world_gi_scale,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            gi_cascades,
        });

        let instance_dynamic_parameters_offset = dynamic_constants
            .push_from_iter(self.instances.iter().map(|inst| inst.dynamic_parameters));

        let triangle_lights_offset: u32 =
            dynamic_constants.push_from_iter(triangle_lights.into_iter());

        self.prev_camera_matrices = Some(frame_desc.camera_matrices);

        rg::renderer::FrameConstantsLayout {
            globals_offset,
            instance_dynamic_parameters_offset,
            triangle_lights_offset,
        }
    }

    pub fn retire_frame(&mut self) {
        self.frame_idx = self.frame_idx.overflowing_add(1).0;
        self.store_prev_mesh_transforms();
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
