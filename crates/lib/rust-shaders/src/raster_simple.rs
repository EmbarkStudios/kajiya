use crate::gbuffer::GBufferData;

use macaw::{Mat3, UVec4, Vec2, Vec3, Vec4, Vec4Swizzles};

use core::mem::size_of;

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

use spirv_std::{
    arch::{ddx_vector, ddy_vector},
    Image, RuntimeArray, Sampler,
};

use rust_shaders_shared::{
    frame_constants::FrameConstants,
    mesh::{InstanceDynamicConstants, InstanceTransform, MaterialDescriptor, MeshDescriptor},
    raster_simple::*,
    util::*,
};

#[spirv(vertex)]
pub fn raster_simple_vs(
    // Pipeline inputs
    #[spirv(vertex_index)] vid: u32,
    #[spirv(instance_index)] instance_index: u32,

    // Descriptors
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    instance_transforms_dyn: &[InstanceTransform],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] meshes: &[MeshDescriptor],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] frame_constants: &FrameConstants,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] vertices: &[u32], // ByteAddressableBuffer
    #[spirv(push_constant)] push_constants: &RasterConstants,

    // Pipeline outputs
    out_color: &mut Vec4,
    out_uv: &mut Vec2,
    out_normal: &mut Vec3,
    #[spirv(flat)] out_material_id: &mut u32,
    out_tangent: &mut Vec3,
    out_bitangent: &mut Vec3,
    out_vs_pos: &mut Vec3,
    out_prev_vs_pos: &mut Vec3,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let mesh: &MeshDescriptor = &meshes[push_constants.mesh_index as usize];

    let (v_pos, v_normal) = load_vertex(vertices, vid * 16 + mesh.vertex_core_offset);

    let v_color = if mesh.vertex_aux_offset != 0 {
        unpack_u32_to_vec4(vertices[(vid * 4 + mesh.vertex_aux_offset) as usize >> 2])
    } else {
        Vec4::ONE
    };

    let v_tangent_packed = if mesh.vertex_tangent_offset != 0 {
        load4f(vertices, vid * 16 + mesh.vertex_tangent_offset)
    } else {
        Vec4::new(1.0, 0.0, 0.0, 1.0)
    };

    let uv: Vec2 = if mesh.vertex_uv_offset != 0 {
        load2f(vertices, vid * 8 + mesh.vertex_uv_offset)
    } else {
        Vec2::ZERO
    };

    let material_id: u32 = vertices[(vid + (mesh.vertex_mat_offset >> 2)) as usize];

    let instance_transform = &instance_transforms_dyn[instance_index as usize];

    let ws_pos = instance_transform.transform.transform_point3(v_pos);

    let prev_ws_pos = if mesh.vertex_prev_core_offset == 0 {
        // The usual flow, just multiply with the current and prev instance transforms.
        instance_transform.prev_transform.transform_point3(v_pos)
    } else {
        // Load the previous position if available (produced by skinning with the last frame's bones).
        instance_transform
            .prev_transform
            .transform_point3(load3f(vertices, vid * 16 + mesh.vertex_prev_core_offset))
    };

    let ws_normal = instance_transform
        .transform
        .transform_vector3(v_normal)
        .normalize();

    let vs_pos: Vec3 = frame_constants
        .view_constants
        .world_to_view
        .transform_point3(ws_pos);
    let cs_pos: Vec4 = frame_constants.view_constants.view_to_sample * vs_pos.extend(1.0);
    let prev_vs_pos: Vec3 = frame_constants
        .view_constants
        .world_to_view
        .transform_point3(prev_ws_pos);

    let tangent = v_tangent_packed.truncate();

    *builtin_pos = cs_pos;
    *out_uv = uv;
    *out_color = v_color;
    *out_normal = ws_normal;
    *out_material_id = material_id;
    *out_tangent = tangent;
    *out_bitangent = (v_normal.cross(tangent) * v_tangent_packed.w).normalize();
    *out_vs_pos = vs_pos;
    *out_prev_vs_pos = prev_vs_pos;
}

#[spirv(fragment)]
pub fn raster_simple_fs(
    // Pipeline inputs

    // Descriptors
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    instance_transforms_dyn: &[InstanceTransform],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] meshes: &[MeshDescriptor],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] frame_constants: &FrameConstants,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] vertices: &[u32], // ByteAddressableBuffer
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)]
    instance_dynamic_parameters_dyn: &[InstanceDynamicConstants],
    #[spirv(descriptor_set = 1, binding = 3)] bindless_textures: &RuntimeArray<
        Image!(2D, type=f32, sampled=true),
    >,
    #[spirv(descriptor_set = 0, binding = 33)] sampler_llr: &Sampler,
    #[spirv(push_constant)] push_constants: &RasterConstants,

    // Pipeline inputs
    in_color: Vec4,
    in_uv: Vec2,
    in_normal: Vec3,
    #[spirv(flat)] in_material_id: u32,
    in_tangent: Vec3,
    in_bitangent: Vec3,
    in_vs_pos: Vec3,
    in_prev_vs_pos: Vec3,

    // Pipeline outputs
    out_geometric_normal: &mut Vec3,
    out_gbuffer: &mut UVec4,
    out_velocity: &mut Vec4,
) {
    let mesh: &MeshDescriptor = &meshes[push_constants.mesh_index as usize];
    let material = MaterialDescriptor::load(
        vertices,
        mesh.vertex_mat_offset + in_material_id * size_of::<MaterialDescriptor>() as u32,
    );

    let albedo_uv = material.transform_uv(in_uv, 0);
    let albedo_tex = unsafe { bindless_textures.index(material.maps.albedo()) };
    let albedo_texel: Vec4 = albedo_tex.sample_bias(*sampler_llr, albedo_uv, -0.5);

    let albedo = albedo_texel.xyz() * material.base_color_mult.xyz() * in_color.xyz();

    let metallic_roughness_uv = material.transform_uv(in_uv, 2);
    let metallic_roughness_tex =
        unsafe { bindless_textures.index(material.maps.metallic_roughness()) };
    let metallic_roughness: Vec4 =
        metallic_roughness_tex.sample_bias(*sampler_llr, metallic_roughness_uv, -0.5);
    let perceptual_roughness = material.roughness_mult * metallic_roughness.y;
    let roughness = perceptual_roughness_to_roughness(perceptual_roughness).clamp(1e-4, 1.0);
    let metalness = metallic_roughness.z * material.metalness_factor;

    let normal_tex = unsafe { bindless_textures.index(material.maps.normal()) };
    let normal_texel: Vec4 = normal_tex.sample_bias(*sampler_llr, in_uv, -0.5);
    let ts_normal = normal_texel.xyz() * 2.0 - 1.0;

    let instance_transform = &instance_transforms_dyn[push_constants.draw_index as usize];

    let mut normal_ws = {
        let normal_os = if in_bitangent.dot(in_bitangent) > 0.0 {
            let tbn = Mat3::from_cols(in_tangent, in_bitangent, in_normal);
            tbn * ts_normal
        } else {
            in_normal
        };

        instance_transform
            .transform
            .transform_vector3(normal_os)
            .normalize()
    };

    // Derive geometric normal from depth
    let geometric_normal_vs = {
        let d1: Vec3 = ddx_vector(in_vs_pos);
        let d2: Vec3 = ddy_vector(in_vs_pos);
        d2.cross(d1).normalize()
    };
    let geometric_normal_ws = frame_constants
        .view_constants
        .view_to_world
        .transform_vector3(geometric_normal_vs);

    // Fix invalid normals
    if normal_ws.dot(geometric_normal_ws) < 0.0 {
        normal_ws *= -1.0;
    }

    let emissive_uv = material.transform_uv(in_uv, 3);
    let emissive_tex = unsafe { bindless_textures.index(material.maps.emissive()) };
    let emissive_texel: Vec4 = emissive_tex.sample_bias(*sampler_llr, emissive_uv, -0.5);
    let emissive = emissive_texel.xyz()
        * material.emissive.xyz()
        * instance_dynamic_parameters_dyn[push_constants.draw_index as usize].emissive_multiplier;

    *out_velocity = (in_prev_vs_pos - in_vs_pos).extend(0.0);
    *out_geometric_normal = geometric_normal_vs * 0.5 + 0.5;

    let gbuffer = GBufferData {
        albedo,
        roughness,
        metalness,
        emissive,
        normal: normal_ws,
    };

    *out_gbuffer = gbuffer.pack();
}
