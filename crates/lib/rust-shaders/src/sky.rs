// not used

#![allow(unreachable_code)]
#![allow(unused_variables)]

use crate::atmosphere::*;
use macaw::{UVec3, Vec2, Vec3};
use rust_shaders_shared::{frame_constants::FrameConstants, util::*};
use spirv_std::Image;

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

fn atmosphere_default(wi: Vec3, light_dir: Vec3, light_color: Vec3) -> Vec3 {
    //return Vec3::ZERO;
    //return Vec3::splat(0.5);

    let world_space_camera_pos = Vec3::ZERO;
    let ray_start = world_space_camera_pos;
    let ray_dir = wi.normalize();
    let ray_length = core::f32::INFINITY;

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
    let uv = (Vec2::new(px.x as f32 + 0.5, px.y as f32 + 0.5)) / 64.0;
    let dir = CUBE_MAP_FACE_ROTATIONS[face as usize] * (uv * 2.0 - Vec2::ONE).extend(-1.0);

    let output = atmosphere_default(
        dir,
        frame_constants.sun_direction.truncate(),
        frame_constants.sun_color_multiplier.truncate() * frame_constants.pre_exposure,
    );
    unsafe {
        output_tex.write(px, output.extend(1.0));
    }
}
