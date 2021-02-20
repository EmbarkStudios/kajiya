#![allow(dead_code)]
#![allow(unused_imports)]

use crate::bytes::into_byte_vec;
use glam::{Mat4, Vec3};
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum TexGamma {
    Linear,
    Srgb,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TexParams {
    pub gamma: TexGamma,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum MeshMaterialMap {
    Asset { path: PathBuf, params: TexParams },
    Placeholder([u8; 4]),
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MeshMaterial {
    pub base_color_mult: [f32; 4],
    pub maps: [u32; 3],
    pub roughness_mult: f32,
    pub metalness_factor: f32,
    pub emissive: [f32; 3],
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
}

fn iter_gltf_node_tree<F: FnMut(&gltf::scene::Node, Mat4)>(
    node: &gltf::scene::Node,
    xform: Mat4,
    f: &mut F,
) {
    let node_xform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let xform = xform * node_xform;

    f(&node, xform);
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
    parent_path: &Path,
) -> (Vec<MeshMaterialMap>, MeshMaterial) {
    let make_asset_path = |path: String| -> PathBuf {
        let mut asset_name: std::path::PathBuf = parent_path.clone().into();
        asset_name.pop();
        asset_name.push(&path);
        asset_name
    };

    let make_material_map = |path: String| -> MeshMaterialMap {
        MeshMaterialMap::Asset {
            path: make_asset_path(path),
            params: TexParams {
                gamma: TexGamma::Linear,
            },
        }
    };

    let albedo_map = mat
        .pbr_metallic_roughness()
        .base_color_texture()
        .and_then(|tex| {
            get_gltf_texture_source(tex.texture()).map(|path: String| -> MeshMaterialMap {
                MeshMaterialMap::Asset {
                    path: make_asset_path(path),
                    params: TexParams {
                        gamma: TexGamma::Srgb,
                    },
                }
            })
        })
        .unwrap_or(MeshMaterialMap::Placeholder([255, 255, 255, 255]));

    let normal_map = mat
        .normal_texture()
        .and_then(|tex| get_gltf_texture_source(tex.texture()).map(make_material_map))
        .unwrap_or(MeshMaterialMap::Placeholder([127, 127, 255, 255]));

    let spec_map = mat
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
        .and_then(|tex| get_gltf_texture_source(tex.texture()).map(make_material_map))
        .unwrap_or(MeshMaterialMap::Placeholder([127, 127, 255, 255]));

    let emissive = if mat.emissive_texture().is_some() {
        [0.0, 0.0, 0.0]
    } else {
        mat.emissive_factor()
    };

    let base_color_mult = mat.pbr_metallic_roughness().base_color_factor();
    let roughness_mult = mat.pbr_metallic_roughness().roughness_factor();
    let metalness_factor = mat.pbr_metallic_roughness().metallic_factor();

    (
        vec![normal_map, spec_map, albedo_map],
        MeshMaterial {
            base_color_mult,
            maps: [0, 1, 2],
            roughness_mult,
            metalness_factor,
            emissive,
        },
    )
}

#[derive(Clone)]
pub struct LoadGltfScene {
    pub path: PathBuf,
    pub scale: f32,
}

impl Hash for LoadGltfScene {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.path.hash(state);
        self.scale.to_ne_bytes().hash(state);
    }
}

#[async_trait]
impl LazyWorker for LoadGltfScene {
    type Output = anyhow::Result<TriangleMesh>;

    async fn run(self, _ctx: RunContext) -> Self::Output {
        let (gltf, buffers, _imgs) = gltf::import(&self.path)
            .with_context(|| format!("Loading GLTF scene from {:?}", self.path))?;

        if let Some(scene) = gltf.default_scene() {
            let mut res: TriangleMesh = TriangleMesh::default();

            let mut process_node = |node: &gltf::scene::Node, xform: Mat4| {
                if let Some(mesh) = node.mesh() {
                    for prim in mesh.primitives() {
                        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

                        let res_material_index = res.materials.len() as u32;

                        {
                            let (mut maps, mut material) =
                                load_gltf_material(&prim.material(), &self.path);

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
                        let mut tangents = if let Some(iter) = reader.read_tangents() {
                            iter.collect::<Vec<_>>()
                        } else {
                            vec![[1.0, 0.0, 0.0, 0.0]; positions.len()]
                        };

                        // Collect uvs (optional)
                        let mut uvs = if let Some(iter) = reader.read_tex_coords(0) {
                            iter.into_f32().collect::<Vec<_>>()
                        } else {
                            vec![[0.0, 0.0]; positions.len()]
                        };

                        // Collect colors (optional)
                        let colors = if let Some(iter) = reader.read_colors(0) {
                            iter.into_rgba_f32().collect::<Vec<_>>()
                        } else {
                            vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
                        };

                        // Collect material ids
                        let mut material_ids = vec![res_material_index; positions.len()];

                        // --------------------------------------------------------
                        // Write it all to the output

                        {
                            let mut indices: Vec<u32>;
                            let base_index = res.positions.len() as u32;

                            if let Some(indices_reader) = reader.read_indices() {
                                indices =
                                    indices_reader.into_u32().map(|i| i + base_index).collect();
                            } else {
                                indices =
                                    (base_index..(base_index + positions.len() as u32)).collect();
                            }

                            // log::info!("Loading a mesh with {} indices", indices.len());

                            res.indices.append(&mut indices);
                            res.tangents.append(&mut tangents);
                            res.material_ids.append(&mut material_ids);
                        }

                        for p in positions {
                            let pos = (xform * Vec3::from(p).extend(1.0)).truncate();
                            res.positions.push(pos.into());
                        }

                        for n in normals {
                            let norm = (xform * Vec3::from(n).extend(0.0)).truncate().normalize();
                            res.normals.push(norm.into());
                        }

                        for c in colors {
                            res.colors.push(c);
                        }

                        res.uvs.append(&mut uvs);
                    }
                }
            };

            let xform = Mat4::from_scale(Vec3::splat(self.scale));
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
    pos: [f32; 3],
    normal: u32,
}

fn pack_unit_direction_11_10_11(x: f32, y: f32, z: f32) -> u32 {
    let x = ((x.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 11u32) - 1u32) as f32) as u32;
    let y = ((y.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 10u32) - 1u32) as f32) as u32;
    let z = ((z.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 11u32) - 1u32) as f32) as u32;

    (z << 21) | (y << 11) | x
}

#[derive(Clone)]
pub struct PackedTriangleMesh {
    pub verts: Vec<PackedVertex>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub material_ids: Vec<u32>,
    pub materials: Vec<MeshMaterial>,
    pub maps: Vec<MeshMaterialMap>,
}

pub fn pack_triangle_mesh(mesh: &TriangleMesh) -> PackedTriangleMesh {
    let mut verts: Vec<PackedVertex> = Vec::with_capacity(mesh.positions.len());

    for (i, pos) in mesh.positions.iter().enumerate() {
        let n = mesh.normals[i];

        verts.push(PackedVertex {
            pos: *pos,
            normal: pack_unit_direction_11_10_11(n[0], n[1], n[2]),
        });
    }

    PackedTriangleMesh {
        verts,
        uvs: mesh.uvs.clone(),
        tangents: mesh.tangents.clone(),
        colors: mesh.colors.clone(),
        indices: mesh.indices.clone(),
        material_ids: mesh.material_ids.clone(),
        materials: mesh.materials.clone(),
        maps: mesh.maps.clone(),
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
