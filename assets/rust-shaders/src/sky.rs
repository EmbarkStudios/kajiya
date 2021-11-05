use crate::atmosphere::*;
use crate::util::*;
use crate::frame_constants::FrameConstants;
use macaw::{const_mat3, vec3, Mat3, UVec3, Vec2, Vec3, Vec4, Vec4Swizzles};
use spirv_std::{Image, Sampler};

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

// TODO: move to cube_map.rs
const CUBE_MAP_FACE_ROTATIONS: [Mat3; 6] = [
    // TODO: Why did we need to swap top and bottom here to match HLSL???
    const_mat3!([0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]), // right
    const_mat3!([0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]),   // left
    const_mat3!([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),   // bottom
    const_mat3!([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),   // top
    const_mat3!([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),  // back
    const_mat3!([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),  // front
];

fn atmosphere_default(wi: Vec3, light_dir: Vec3) -> Vec3 {
    let world_space_camera_pos = Vec3::ZERO;
    let ray_start = world_space_camera_pos;
    let ray_dir = wi.normalize();
    let ray_length = core::f32::INFINITY;

    let light_color = Vec3::ONE;

    let mut transmittance = Vec3::ZERO;
    integrate_scattering(
        ray_start,
        ray_dir,
        ray_length,
        light_dir,
        light_color,
        &mut transmittance,
    )
}

#[spirv(compute(threads(8, 8, 1)))]
pub fn comp_sky_cube_cs(
    #[spirv(descriptor_set = 0, binding = 0)] output_tex: &Image!(2D, format=rgba16f, sampled=false, arrayed=true),
    #[spirv(uniform, descriptor_set = 2, binding = 0)] frame_constants: &FrameConstants,
    #[spirv(global_invocation_id)] px: UVec3,
) {
    let face = px.z;
    let uv = (Vec2::new(px.x as f32 + 0.5, px.y as f32 + 0.5)) / 32.0;
    let dir = CUBE_MAP_FACE_ROTATIONS[face as usize] * (uv * 2.0 - Vec2::ONE).extend(-1.0);

    let output = atmosphere_default(dir, frame_constants.sun_direction.truncate());
    unsafe {
        output_tex.write(px, output.extend(1.0));
    }
}

pub struct PrefilterConstants {
    face_size: u32,
    roughness: f32,
}

fn importance_sample_ggx(u: Vec2, n: Vec3, roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let phi = core::f32::consts::TAU * u.x;
    let cos_theta = ((1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    // Tangent space H
    let h = vec3(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);

    // Generate basis
    let up = if n.z.abs() < 0.999 {
        vec3(0.0, 0.0, 1.0)
    } else {
        vec3(1.0, 0.0, 0.0)
    };
    let tangent = up.cross(n).normalize();
    let bitangent = n.cross(tangent);

    // World space H
    (tangent * h.x + bitangent * h.y + n * h.z).normalize()
}

#[spirv(compute(threads(8, 8, 1)))]
pub fn prefilter_sky_cube(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(cube, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] output_tex: &Image!(2D, format=rgba16f, sampled=false, arrayed=true),
    #[spirv(uniform, descriptor_set = 0, binding = 2)] constants: &PrefilterConstants,
    #[spirv(descriptor_set = 0, binding = 33)] sampler_llr: &Sampler,
    #[spirv(global_invocation_id)] px: UVec3,
) {
    let face = px.z as usize;
    let uv = (Vec2::new(px.x as f32 + 0.5, px.y as f32 + 0.5)) / constants.face_size as f32;

    let n = (CUBE_MAP_FACE_ROTATIONS[face] * (uv * 2.0 - Vec2::ONE).extend(-1.0)).normalize();

    let sample_count = 96u32;
    let mut total_weight = 0.0f32;
    let mut color = Vec3::ZERO;

    for i in 0..sample_count {
        let urand: Vec2 = hammersley(i, sample_count);

        let h = importance_sample_ggx(urand, n, constants.roughness);
        let l = (2.0 * n.dot(h) * h - n).normalize();
        let n_dot_l = n.dot(l).max(0.0);

        if n_dot_l > 0.0 {
            let sample: Vec4 = input_tex.sample_by_lod(*sampler_llr, l, 0.0);

            color += sample.xyz() * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    let output = color / total_weight as f32;
    unsafe {
        output_tex.write(px, output.extend(1.0));
    }
}