#![allow(dead_code)]
#![allow(unused_imports)]

use byteorder::{ByteOrder, NativeEndian, WriteBytesExt};
use glam::{Mat4, Quat, Vec3, Vec4};
use gltf::texture::TextureTransform;
use kajiya_backend::bytes::into_byte_vec;
/*use render_core::{
    constants::MAX_VERTEX_STREAMS,
    device::RenderDevice,
    state::{build, RenderBindingBuffer},
    types::{
        RayTracingGeometryDesc, RayTracingGeometryPart, RayTracingGeometryType, RenderBindFlags,
        RenderBufferDesc, RenderDrawBindingSetDesc, RenderFormat, RenderResourceType,
        RenderShaderViewsDesc,
    },
};*/
use anyhow::Context as _;
use std::{
    hash::Hash,
    mem::size_of,
    path::{Path, PathBuf},
};
use turbosloth::*;

use crate::image::ImageSource;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum TexGamma {
    Linear,
    Srgb,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum TexCompressionMode {
    None,
    Rgba,
    Rg,
}

impl TexCompressionMode {
    pub fn supports_alpha(&self) -> bool {
        match self {
            TexCompressionMode::None => true,
            TexCompressionMode::Rgba => true,
            TexCompressionMode::Rg => false,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TexParams {
    pub gamma: TexGamma,
    pub use_mips: bool,
    pub compression: TexCompressionMode,
    pub channel_swizzle: Option<[usize; 4]>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum MeshMaterialMap {
    Image {
        source: ImageSource,
        params: TexParams,
    },
    Placeholder([u8; 4]),
}

pub struct MeshMaterialFlags;
impl MeshMaterialFlags {
    pub const MESH_MATERIAL_FLAG_EMISSIVE_USED_AS_LIGHT: u32 = 1;
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MeshMaterial {
    pub base_color_mult: [f32; 4],
    pub maps: [u32; 4],
    pub roughness_mult: f32,
    pub metalness_factor: f32,
    pub emissive: [f32; 3],
    pub flags: u32,
    pub map_transforms: [[f32; 6]; 4],
}

#[derive(Clone, Default)]
pub struct TriangleMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub material_ids: Vec<u32>, // per index, but can be flat shaded
    pub indices: Vec<u32>,
    pub materials: Vec<MeshMaterial>, // global
    pub maps: Vec<MeshMaterialMap>,   // global
    pub images: Vec<ImageSource>,
}

fn iter_gltf_node_tree<F: FnMut(&gltf::scene::Node, Mat4)>(
    node: &gltf::scene::Node,
    xform: Mat4,
    f: &mut F,
) {
    let node_xform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let xform = xform * node_xform;

    f(node, xform);
    for child in node.children() {
        iter_gltf_node_tree(&child, xform, f);
    }
}

fn get_gltf_texture_source(tex: gltf::texture::Texture) -> Option<String> {
    match tex.source().source() {
        gltf::image::Source::Uri { uri, .. } => Some(uri.to_string()),
        _ => None,
    }
}

fn load_gltf_material(
    mat: &gltf::material::Material,
    document_images: &[ImageSource],
) -> (Vec<MeshMaterialMap>, MeshMaterial) {
    const DEFAULT_MAP_TRANSFORM: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut map_transforms: [[f32; 6]; 4] = [DEFAULT_MAP_TRANSFORM; 4];

    fn texture_transform_to_matrix(xform: Option<TextureTransform>) -> [f32; 6] {
        if let Some(xform) = xform {
            let r = xform.rotation();
            let s = xform.scale();
            let o = xform.offset();

            [
                r.cos() * s[0],
                r.sin() * s[1],
                -r.sin() * s[0],
                r.cos() * s[1],
                o[0],
                o[1],
            ]
        } else {
            DEFAULT_MAP_TRANSFORM
        }
    }

    let (albedo_map, albedo_map_transform) = mat
        .pbr_metallic_roughness()
        .base_color_texture()
        .or_else(|| mat.pbr_specular_glossiness()?.diffuse_texture())
        .map_or(
            (
                MeshMaterialMap::Placeholder([255, 255, 255, 255]),
                DEFAULT_MAP_TRANSFORM,
            ),
            |tex| {
                let transform = texture_transform_to_matrix(tex.texture_transform());

                (
                    MeshMaterialMap::Image {
                        source: document_images[tex.texture().source().index()].clone(),
                        params: TexParams {
                            gamma: TexGamma::Srgb,
                            use_mips: true,
                            compression: TexCompressionMode::Rgba,
                            channel_swizzle: None,
                        },
                    },
                    transform,
                )
            },
        );

    map_transforms[0] = albedo_map_transform;

    // TODO: add texture transform to the normal map in the `gltf` crate
    let normal_map =
        mat.normal_texture()
            .map_or(MeshMaterialMap::Placeholder([127, 127, 255, 255]), |tex| {
                MeshMaterialMap::Image {
                    source: document_images[tex.texture().source().index()].clone(),
                    params: TexParams {
                        gamma: TexGamma::Linear,
                        use_mips: true,
                        compression: TexCompressionMode::Rg,
                        channel_swizzle: None,
                    },
                }
            });

    let (spec_map, spec_map_transform) = mat
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
        .map_or_else(
            || {
                let roughness = 255;
                let metalness = 255;
                (
                    MeshMaterialMap::Placeholder([roughness, metalness, 127, 255]),
                    DEFAULT_MAP_TRANSFORM,
                )
            },
            |tex| {
                (
                    MeshMaterialMap::Image {
                        source: document_images[tex.texture().source().index()].clone(),
                        params: TexParams {
                            gamma: TexGamma::Linear,
                            use_mips: true,
                            compression: TexCompressionMode::Rg,
                            channel_swizzle: Some([1, 2, 0, 3]),
                        },
                    },
                    texture_transform_to_matrix(tex.texture_transform()),
                )
            },
        );

    map_transforms[2] = spec_map_transform;

    let mut emissive_map = MeshMaterialMap::Placeholder([255, 255, 255, 255]);
    if let Some(tex) = mat.emissive_texture() {
        map_transforms[3] = texture_transform_to_matrix(tex.texture_transform());
        emissive_map = MeshMaterialMap::Image {
            source: document_images[tex.texture().source().index()].clone(),
            params: TexParams {
                gamma: TexGamma::Srgb,
                use_mips: true,
                compression: TexCompressionMode::Rgba,
                channel_swizzle: None,
            },
        }
    }

    let emissive = mat.emissive_factor();

    let base_color_mult = mat.pbr_metallic_roughness().base_color_factor();
    let roughness_mult = mat.pbr_metallic_roughness().roughness_factor();
    let metalness_factor = mat.pbr_metallic_roughness().metallic_factor();

    //mata.normal_texture().and_then(|tex| tex.transform())

    (
        vec![normal_map, spec_map, albedo_map, emissive_map],
        MeshMaterial {
            base_color_mult,
            maps: [0, 1, 2, 3],
            roughness_mult,
            metalness_factor,
            emissive,
            flags: 0,
            map_transforms,
        },
    )
}

#[derive(Clone)]
pub struct LoadGltfScene {
    pub path: PathBuf,
    pub scale: f32,
    pub rotation: Quat,
}

impl Hash for LoadGltfScene {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.path.hash(state);
        self.scale.to_ne_bytes().hash(state);
        self.rotation.x.to_ne_bytes().hash(state);
        self.rotation.y.to_ne_bytes().hash(state);
        self.rotation.z.to_ne_bytes().hash(state);
        self.rotation.w.to_ne_bytes().hash(state);
    }
}

#[async_trait]
impl LazyWorker for LoadGltfScene {
    type Output = anyhow::Result<TriangleMesh>;

    async fn run(self, _ctx: RunContext) -> Self::Output {
        let (gltf, buffers, imgs) = crate::import_gltf::import(&self.path)
            .with_context(|| format!("Loading GLTF scene from {:?}", self.path))?;

        if let Some(scene) = gltf.default_scene().or_else(|| gltf.scenes().next()) {
            let mut res: TriangleMesh = TriangleMesh::default();

            let mut process_node = |node: &gltf::scene::Node, xform: Mat4| {
                if let Some(mesh) = node.mesh() {
                    let flip_winding_order = xform.determinant() < 0.0;

                    for prim in mesh.primitives() {
                        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

                        let res_material_index = res.materials.len() as u32;

                        {
                            let (mut maps, mut material) =
                                load_gltf_material(&prim.material(), imgs.as_slice());

                            let map_base = res.maps.len() as u32;
                            for id in material.maps.iter_mut() {
                                *id += map_base;
                            }

                            res.materials.push(material);
                            res.maps.append(&mut maps);
                        }

                        // Collect positions (required)
                        let positions = if let Some(iter) = reader.read_positions() {
                            iter.collect::<Vec<_>>()
                        } else {
                            return;
                        };

                        // Collect normals (required)
                        let normals = if let Some(iter) = reader.read_normals() {
                            iter.collect::<Vec<_>>()
                        } else {
                            return;
                        };

                        // Collect tangents (optional)
                        let (mut tangents, tangents_found) =
                            if let Some(iter) = reader.read_tangents() {
                                (iter.collect::<Vec<_>>(), true)
                            } else {
                                (vec![[1.0, 0.0, 0.0, 0.0]; positions.len()], false)
                            };

                        // Collect uvs (optional)
                        let (mut uvs, uvs_found) = if let Some(iter) = reader.read_tex_coords(0) {
                            (iter.into_f32().collect::<Vec<_>>(), true)
                        } else {
                            (vec![[0.0, 0.0]; positions.len()], false)
                        };

                        // Collect colors (optional)
                        let mut colors = if let Some(iter) = reader.read_colors(0) {
                            iter.into_rgba_f32().collect::<Vec<_>>()
                        } else {
                            vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
                        };

                        // Collect material ids
                        let mut material_ids = vec![res_material_index; positions.len()];

                        // Collect indices
                        let mut indices: Vec<u32>;
                        {
                            if let Some(indices_reader) = reader.read_indices() {
                                indices = indices_reader.into_u32().collect();
                            } else {
                                if positions.is_empty() {
                                    return;
                                }

                                match prim.mode() {
                                    gltf::mesh::Mode::Triangles => {
                                        indices = (0..positions.len() as u32).collect();
                                    }
                                    _ => {
                                        panic!(
                                            "Primitive mode {:?} not supported yet",
                                            prim.mode()
                                        );
                                    }
                                }
                            }

                            if flip_winding_order {
                                for tri in indices.chunks_exact_mut(3) {
                                    tri.swap(0, 2);
                                }
                            }
                        }

                        if !tangents_found && uvs_found {
                            log::trace!(
                                "Mesh had UVs but no tangents. Calculating the tangents..."
                            );

                            mikktspace::generate_tangents(&mut TangentCalcContext {
                                indices: indices.as_slice(),
                                positions: positions.as_slice(),
                                normals: normals.as_slice(),
                                uvs: uvs.as_slice(),
                                tangents: tangents.as_mut_slice(),
                            });
                        }

                        // --------------------------------------------------------
                        // Write it all to the output

                        {
                            // log::info!("Loading a mesh with {} indices", indices.len());
                            let base_index = res.positions.len() as u32;
                            for i in &mut indices {
                                *i += base_index;
                            }

                            res.indices.append(&mut indices);
                            res.colors.append(&mut colors);
                            res.material_ids.append(&mut material_ids);
                        }

                        for v in positions {
                            let pos = (xform * Vec3::from(v).extend(1.0)).truncate();
                            res.positions.push(pos.into());
                        }

                        for v in normals {
                            let norm = (xform * Vec3::from(v).extend(0.0)).truncate().normalize();
                            res.normals.push(norm.into());
                        }

                        for v in tangents {
                            let v = Vec4::from(v);
                            let t = (xform * v.truncate().extend(0.0)).truncate().normalize();
                            res.tangents.push(
                                t.extend(v.w * if flip_winding_order { -1.0 } else { 1.0 })
                                    .into(),
                            );
                        }

                        res.uvs.append(&mut uvs);
                    }
                }
            };

            let xform = Mat4::from_scale_rotation_translation(
                Vec3::splat(self.scale),
                self.rotation,
                Vec3::ZERO,
            );
            for node in scene.nodes() {
                iter_gltf_node_tree(&node, xform, &mut process_node);
            }

            Ok(res)
        } else {
            Err(anyhow::anyhow!("No default scene found in gltf"))
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PackedVertex {
    pub pos: [f32; 3],
    normal: u32,
}

fn pack_unit_direction_11_10_11(x: f32, y: f32, z: f32) -> u32 {
    let x = ((x.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 11u32) - 1u32) as f32) as u32;
    let y = ((y.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 10u32) - 1u32) as f32) as u32;
    let z = ((z.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 11u32) - 1u32) as f32) as u32;

    (z << 21) | (y << 11) | x
}

#[repr(packed)]
pub struct FlatVec<T> {
    len: u64,
    offset: u64,
    marker: std::marker::PhantomData<T>,
}

impl<T> std::borrow::Borrow<[T]> for FlatVec<T> {
    fn borrow(&self) -> &[T] {
        unsafe {
            let data = (self as *const Self as *const u8).add(self.offset as usize);
            std::slice::from_raw_parts(data as *const T, self.len as usize)
        }
    }
}

impl<T> std::ops::Index<usize> for FlatVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T> FlatVec<T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            let data = (std::ptr::addr_of!(self.offset) as *const u64 as *const u8)
                .add(self.offset as usize);
            std::slice::from_raw_parts(data as *const T, self.len as usize)
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        unsafe {
            let data = (std::ptr::addr_of!(self.offset) as *const u64 as *const u8)
                .add(self.offset as usize);
            data as *const T
        }
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.as_slice().iter()
    }
}

pub fn flatten_vec_header(writer: &mut Vec<u8>, len: usize) -> usize {
    writer
        .write_u64::<byteorder::NativeEndian>(len as u64)
        .unwrap();
    writer.write_u64::<byteorder::NativeEndian>(0u64).unwrap();
    writer.len() - 8
}

pub fn flatten_plain_field<T: Copy + Sized>(writer: &mut impl std::io::Write, data: &T) {
    writer
        .write_all(unsafe {
            std::slice::from_raw_parts(data as *const T as *const u8, std::mem::size_of::<T>())
        })
        .unwrap();
}

pub fn flatten_bytes(writer: &mut impl std::io::Write, data: &[u8]) {
    writer.write_all(data).unwrap();
}

pub struct DeferredBlob {
    pub fixup_addr: usize, // offset within parent
    pub nested: FlattenCtx,
}

#[derive(Default)]
pub struct FlattenCtx {
    pub section_idx: Option<usize>,
    pub bytes: Vec<u8>,
    pub deferred: Vec<DeferredBlob>,
}

impl FlattenCtx {
    fn allocate_section_indices(&mut self) {
        let mut counter = 0;
        self.allocate_section_indices_impl(&mut counter);
    }

    fn allocate_section_indices_impl(&mut self, counter: &mut usize) {
        self.section_idx = Some(*counter);
        *counter += 1;

        for child in &mut self.deferred {
            child.nested.allocate_section_indices_impl(counter);
        }
    }

    fn finish(mut self, writer: &mut impl std::io::Write) {
        self.allocate_section_indices();

        type FixupAddr = usize;
        type SectionIdx = usize;

        struct Section {
            bytes: Vec<u8>,
            fixups: Vec<(FixupAddr, SectionIdx)>,
        }

        // Build flattened sections over the nested structures
        let mut sections: Vec<Section> = Vec::new();

        let mut ctx_list = vec![self];
        while !ctx_list.is_empty() {
            let mut next_ctx_list: Vec<Self> = vec![];

            for ctx in ctx_list {
                sections.push(Section {
                    bytes: ctx.bytes,
                    fixups: ctx
                        .deferred
                        .iter()
                        .map(|deferred| (deferred.fixup_addr, deferred.nested.section_idx.unwrap()))
                        .collect(),
                });

                for deferred in ctx.deferred {
                    next_ctx_list.push(deferred.nested);
                }
            }

            ctx_list = next_ctx_list;
        }

        // Lay out the sections
        let mut total_bytes = 0usize;
        let section_base_addr: Vec<usize> = sections
            .iter()
            .map(|s| {
                let base_addr = total_bytes;
                total_bytes += s.bytes.len();
                base_addr
            })
            .collect();

        // Apply fixups
        for (section, &section_addr) in sections.iter_mut().zip(&section_base_addr) {
            for &(fixup_addr, target_section) in &section.fixups {
                // Apply the fixup of the structure which was pointing at this deferred blob
                let fixup_target: u64 = section_base_addr[target_section] as u64;

                // The fixup is relative to the `offset` field itself
                let fixup_relative = fixup_target - (fixup_addr + section_addr) as u64;

                section.bytes[fixup_addr..fixup_addr + 8]
                    .copy_from_slice(&fixup_relative.to_ne_bytes());
            }
        }

        // Write sections out
        for section in sections {
            writer.write_all(section.bytes.as_slice()).unwrap();
        }
    }
}

macro_rules! def_asset {
    // Vector
    (@proto_ty Vec($($type:tt)+)) => {
        Vec<def_asset!(@proto_ty $($type)+ )>
    };
    (@flat_ty Vec($($type:tt)+)) => {
        FlatVec<def_asset!(@flat_ty $($type)+ )>
    };
    (@flatten $output:expr; $field:expr; Vec($($type:tt)+)) => {
        let fixup_addr = flatten_vec_header(&mut $output.bytes, $field.len());
        let mut nested = FlattenCtx::default();
        for item in $field.iter() {
            def_asset!(@flatten &mut nested; item; $($type)+ );
        }
        $output.deferred.push(DeferredBlob {
            fixup_addr,
            nested,
        });
    };

    // Bytes
    (@proto_ty Bytes) => {
        bytes::Bytes
    };
    (@flat_ty Bytes) => {
        FlatVec<u8>
    };
    (@flatten $output:expr; $field:expr; Bytes) => {
        let fixup_addr = flatten_vec_header(&mut $output.bytes, $field.len());
        let mut nested = FlattenCtx::default();
        flatten_bytes(&mut $output.bytes, $field);
        $output.deferred.push(DeferredBlob {
            fixup_addr,
            nested,
        });
    };

    // Asset
    (@proto_ty Asset($($type:tt)+)) => {
        Lazy<def_asset!(@proto_ty $($type)+ ::Proto)>
    };
    (@flat_ty Asset($($type:tt)+)) => {
        AssetRef<def_asset!(@flat_ty $($type)+ ::Flat)>
    };
    (@flatten $output:expr; $field:expr; Asset($($type:tt)+)) => {
        let asset_ref: AssetRef<$($type ::Flat)+> = AssetRef {
            identity: $field.identity(),
            marker: std::marker::PhantomData,
        };
        flatten_plain_field(&mut $output.bytes, &asset_ref)
    };


    // Plain type
    (@proto_ty $($type:tt)+) => {
        $($type)+
    };
    (@flat_ty $($type:tt)+) => {
        $($type)+
    };
    (@flatten $output:expr; $field:expr; $($type:tt)+) => {
        flatten_plain_field(&mut $output.bytes, $field)
    };

    (
        $(
            #[derive($($derive:tt)+)]
        )?
        $struct_name:ident {
            $(
                $name:ident { $($type:tt)+ }
            )+
        }
    ) => {
        #[allow(non_snake_case)]
        pub mod $struct_name {
            use super::*;

            $(#[derive($($derive)+)])?
            pub struct Proto {
                $(
                    pub $name: def_asset!(@proto_ty $($type)+ ),
                )*
            }

            #[repr(packed)]
            pub struct Flat {
                $(
                    pub $name: def_asset!(@flat_ty $($type)+ ),
                )*
            }

            impl Proto {
                pub fn flatten_into(&self, writer: &mut impl std::io::Write) {
                    let mut output = FlattenCtx {
                        bytes: Vec::new(),
                        deferred: Default::default(),
                        section_idx: None,
                    };

                    $(
                        def_asset!(@flatten &mut output; &self.$name; $($type)+ );
                    )*

                    output.finish(writer)
                }
            }
        }
    };
}

#[repr(C)]
pub struct AssetRef<T> {
    identity: u64,
    marker: std::marker::PhantomData<fn(&T)>,
}
impl<T> AssetRef<T> {
    pub fn identity(&self) -> u64 {
        self.identity
    }
}
impl<T> Clone for AssetRef<T> {
    fn clone(&self) -> Self {
        Self {
            identity: self.identity,
            marker: std::marker::PhantomData,
        }
    }
}
impl<T> Copy for AssetRef<T> {}
impl<T> PartialEq for AssetRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}
impl<T> Eq for AssetRef<T> {}
impl<T> Hash for AssetRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.identity)
    }
}
impl<T> PartialOrd for AssetRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.identity.partial_cmp(&other.identity)
    }
}
impl<T> Ord for AssetRef<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.identity.cmp(&other.identity)
    }
}

// TODO: use `rkyv` instead
def_asset! {
    GpuImage {
        format { kajiya_backend::ash::vk::Format }
        extent { [u32; 3] }
        mips { Vec(Vec(u8)) }
    }
}

// TODO: use `rkyv` instead
def_asset! {
    #[derive(Clone)]
    PackedTriMesh {
        verts { Vec(PackedVertex) }
        uvs { Vec([f32; 2]) }
        tangents { Vec([f32; 4]) }
        colors { Vec([f32; 4]) }
        indices { Vec(u32) }
        material_ids { Vec(u32) }
        materials { Vec(MeshMaterial) }
        maps { Vec(Asset(GpuImage)) }
    }
}

/*#[derive(Clone)]
pub struct PackedTriangleMesh {
    pub verts: Vec<PackedVertex>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub material_ids: Vec<u32>,
    pub materials: Vec<MeshMaterial>,
    pub maps: Vec<MeshMaterialMap>,
}*/

pub type PackedTriangleMesh = PackedTriMesh::Proto;

pub fn pack_triangle_mesh(mesh: &TriangleMesh) -> PackedTriangleMesh {
    let mut verts: Vec<PackedVertex> = Vec::with_capacity(mesh.positions.len());

    for (i, pos) in mesh.positions.iter().enumerate() {
        let n = mesh.normals[i];

        verts.push(PackedVertex {
            pos: *pos,
            normal: pack_unit_direction_11_10_11(n[0], n[1], n[2]),
        });
    }

    let maps = mesh
        .maps
        .iter()
        .map(|map| {
            let (image, params) = match map {
                MeshMaterialMap::Image { source, params } => (
                    super::image::LoadImage::new(source).unwrap().into_lazy(),
                    *params,
                ),
                MeshMaterialMap::Placeholder(values) => (
                    super::image::CreatePlaceholderImage::new(*values).into_lazy(),
                    TexParams {
                        gamma: crate::mesh::TexGamma::Linear,
                        use_mips: false,
                        compression: TexCompressionMode::None,
                        channel_swizzle: None,
                    },
                ),
            };

            crate::image::CreateGpuImage { image, params }.into_lazy()
        })
        .collect();

    PackedTriangleMesh {
        verts,
        uvs: mesh.uvs.clone(),
        tangents: mesh.tangents.clone(),
        colors: mesh.colors.clone(),
        indices: mesh.indices.clone(),
        material_ids: mesh.material_ids.clone(),
        materials: mesh.materials.clone(),
        maps,
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
struct GpuMaterial {
    base_color_mult: [f32; 4],
    maps: [u32; 4],
}

impl Default for GpuMaterial {
    fn default() -> Self {
        Self {
            base_color_mult: [0.0f32; 4],
            maps: [0; 4],
        }
    }
}

struct TangentCalcContext<'a> {
    indices: &'a [u32],
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    uvs: &'a [[f32; 2]],
    tangents: &'a mut [[f32; 4]],
}

impl<'a> mikktspace::Geometry for TangentCalcContext<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.indices[face * 3 + vert] as usize]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.indices[face * 3 + vert] as usize]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.indices[face * 3 + vert] as usize]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.tangents[self.indices[face * 3 + vert] as usize] = tangent;
    }
}
