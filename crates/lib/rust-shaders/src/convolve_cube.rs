// not used

use macaw::{Mat3, UVec3, Vec2, Vec3, Vec4};
use rust_shaders_shared::util::*;
use spirv_std::{Image, Sampler};

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[repr(C)]
pub struct Constants {
    pub face_width: u32,
}

#[spirv(compute(threads(8, 8, 1)))]
pub fn convolve_cube_cs(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(cube, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] output_tex: &Image!(2D, format=rgba16f, sampled=false, arrayed=true),
    #[spirv(uniform, descriptor_set = 0, binding = 2)] constants: &Constants,
    #[spirv(descriptor_set = 0, binding = 33)] sampler_llr: &Sampler,
    #[spirv(global_invocation_id)] px: UVec3,
) {
    let face = px.z as usize;
    let uv = (Vec2::new(px.x as f32 + 0.5, px.y as f32 + 0.5)) / constants.face_width as f32;

    let output_dir =
        (CUBE_MAP_FACE_ROTATIONS[face] * (uv * 2.0 - Vec2::ONE).extend(-1.0)).normalize();
    let basis: Mat3 = build_orthonormal_basis(output_dir);

    let sample_count: u32 = 256;

    let mut result: Vec4 = Vec4::ZERO;
    let mut i = 0;
    while i < sample_count {
        let urand: Vec2 = hammersley(i, sample_count);
        let input_dir: Vec3 = basis * uniform_sample_cone(urand, 0.85);
        let sample: Vec4 = input_tex.sample_by_lod(*sampler_llr, input_dir, 0.0);
        result += sample;
        i += 1;
    }

    unsafe {
        output_tex.write(px, result / sample_count as f32);
    }
}
