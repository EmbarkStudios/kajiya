// Rust-GPU port of `motion_blur.hlsl` by Viktor Zoutman

use crate::{
    frame_constants::FrameConstants,
    util::{depth_to_view_z, get_uv_u},
};
use spirv_std::{
    Image, Sampler,
};
use macaw::{uvec2, vec2, IVec2, IVec3, UVec2, UVec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[repr(C)]
pub struct Constants {
    depth_tex_size: Vec4,
    output_tex_size: Vec4,
    motion_blur_scale: f32,
}

fn depth_cmp(center_depth: f32, sample_depth: f32, depth_scale: f32) -> Vec2 {
    // Clamp NaN's to a value since sample_depth - center_depth can both be INF
    (Vec2::splat(0.5) + vec2(depth_scale, -depth_scale) * (sample_depth - center_depth))
        .clamp(Vec2::ZERO, Vec2::ONE)
}

fn spread_cmp(offset_len: f32, spread_len: Vec2) -> Vec2 {
    return (spread_len - Vec2::splat(offset_len + 1.0)).clamp(Vec2::ZERO, Vec2::ONE);
}

fn sample_weight(
    center_depth: f32,
    sample_depth: f32,
    offset_len: f32,
    center_spread_len: f32,
    sample_spread_len: f32,
    depth_scale: f32,
) -> f32 {
    let dc = depth_cmp(center_depth, sample_depth, depth_scale);
    let sc = spread_cmp(offset_len, vec2(center_spread_len, sample_spread_len));
    dc.dot(sc)
}

// Workaround for https://github.com/EmbarkStudios/rust-gpu/issues/699
fn clamp_uvec2(a: UVec2, min: UVec2, max: UVec2) -> UVec2 {
    uvec2(a.x.clamp(min.x, max.x), a.y.clamp(min.y, max.y))
}

#[spirv(compute(threads(8, 8)))]
pub fn motion_blur(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] velocity_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] tile_velocity_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] depth_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(descriptor_set = 0, binding = 32)] sampler_lnc: &Sampler,
    #[spirv(descriptor_set = 0, binding = 33)] sampler_nnc: &Sampler,
    //#[spirv(descriptor_set = 0, binding = 33)] sampler_nnc: &Sampler,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] constants: &Constants,
    #[spirv(uniform, descriptor_set = 2, binding = 0)] frame_constants: &FrameConstants,
    #[spirv(global_invocation_id)] px: UVec3,
) {
    let uv = get_uv_u(px.xy(), constants.output_tex_size);
    let blur_scale = 0.5 * constants.motion_blur_scale;
    let depth_tex_size = constants.depth_tex_size.xy();
    let output_tex_size = constants.output_tex_size.xy();

    // Scramble tile coordinates to diffuse the tile quantization in noise
    let mut noise1: i32;
    let mut tile_offset = px.xy().as_ivec2();
    {
        tile_offset.x += tile_offset.x << 4;
        tile_offset.x ^= tile_offset.x >> 6;
        tile_offset.y += tile_offset.x << 1;
        tile_offset.y += tile_offset.y << 6;
        tile_offset.y ^= tile_offset.y >> 2;
        tile_offset.x ^= tile_offset.y;
        noise1 = tile_offset.x ^ (tile_offset.y << 1);
        tile_offset.x &= 31;
        tile_offset.y &= 31;
        tile_offset -= IVec2::splat(15);
        noise1 &= 31;
        noise1 -= 15;
    }

    let velocity_tile_coord = ((uv * depth_tex_size) + tile_offset.as_vec2()).as_uvec2();
    let tile_velocity: Vec4 = tile_velocity_tex.fetch(
        clamp_uvec2(
            velocity_tile_coord,
            UVec2::splat(0),
            depth_tex_size.as_uvec2() - UVec2::splat(1),
        ) / 16,
    );
    let tile_velocity = blur_scale * tile_velocity.xy();

    let kernel_width = 4;
    let noise = 0.5 * noise1 as f32 / 15.0;

    let center_offset_len = noise / kernel_width as f32 * 0.5;
    let center_uv = uv + tile_velocity * center_offset_len;

    //float3 center_color = texelFetch(inputImage, px, 0).rgb;
    let center_color: Vec4 = input_tex.fetch(clamp_uvec2(
        (center_uv * output_tex_size).as_uvec2(),
        UVec2::splat(0),
        output_tex_size.as_uvec2() - UVec2::splat(1),
    ));
    let center_color = center_color.xyz();
    let center_depth: Vec4 = depth_tex.sample_by_lod(*sampler_nnc, center_uv, 0.0);
    let center_depth = -depth_to_view_z(center_depth.x, frame_constants);
    let center_velocity: Vec4 = velocity_tex.sample_by_lod(*sampler_lnc, center_uv, 0.0);
    let center_velocity = blur_scale * center_velocity.xy();
    let center_velocity_px = center_velocity * depth_tex_size;

    let soft_z = 16.0;
    let mut sum = Vec4::splat(0.0);
    let mut sample_count = 1.0;
    if tile_velocity.length() > 0.0 {
        for i in 1..kernel_width {
            let offset_len0 = (i as f32 + noise) / kernel_width as f32 * 0.5;
            let offset_len1 = (-i as f32 + noise) / kernel_width as f32 * 0.5;
            let uv0 = uv + tile_velocity * offset_len0;
            let uv1 = uv + tile_velocity * offset_len1;
            let px0 = (uv0 * depth_tex_size).as_uvec2();
            let px1 = (uv1 * depth_tex_size).as_uvec2();

            let d0: Vec4 = depth_tex.fetch(px0);
            let d1: Vec4 = depth_tex.fetch(px1);
            let d0 = -depth_to_view_z(d0.x, frame_constants);
            let d1 = -depth_to_view_z(d1.x, frame_constants);
            let v0: Vec4 = velocity_tex.fetch(px0);
            let v1: Vec4 = velocity_tex.fetch(px1);
            let v0 = (blur_scale * v0.xy() * depth_tex_size).length();
            let v1 = (blur_scale * v1.xy() * depth_tex_size).length();

            let mut weight0 = sample_weight(
                center_depth,
                d0,
                ((uv0 - uv) * depth_tex_size).length(),
                center_velocity_px.length(),
                v0,
                soft_z,
            );
            let mut weight1 = sample_weight(
                center_depth,
                d1,
                ((uv1 - uv) * depth_tex_size).length(),
                center_velocity_px.length(),
                v1,
                soft_z,
            );

            let mirror = (d0 > d1, v1 > v0);
            weight0 = if mirror.0 && mirror.1 {
                weight1
            } else {
                weight0
            };
            weight1 = if mirror.0 || mirror.1 {
                weight1
            } else {
                weight0
            };

            let valid0 = (uv0 == uv0.clamp(Vec2::ZERO, Vec2::ONE)) as i32 as f32;
            let valid1 = (uv1 == uv1.clamp(Vec2::ZERO, Vec2::ONE)) as i32 as f32;

            weight0 *= valid0;
            weight1 *= valid1;
            sample_count += valid0 + valid1;

            sum += {
                let mut c: Vec4 = input_tex.sample_by_lod(*sampler_lnc, uv0, 1.0);
                c.w = 1.0;
                c * weight0
            };
            sum += {
                let mut c: Vec4 = input_tex.sample_by_lod(*sampler_lnc, uv1, 1.0);
                c.w = 1.0;
                c * weight1
            };
        }

        //sum *= 1.0 / (kernel_width * 2 + 1);
        sum *= 1.0 / sample_count;
    }

    let result = sum.xyz() + (1.0 - sum.w) * center_color;
    unsafe {
        output_tex.write(px.truncate(), result);
    }
}

#[spirv(compute(threads(8, 8)))]
pub fn velocity_reduce_x(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(global_invocation_id)] px: UVec3,
) {
    let mut largest_velocity = Vec3::ZERO;

    for x in 0..16 {
        let v: Vec4 = input_tex.fetch(px.xy() * uvec2(16, 1) + uvec2(x, 0));
        let v = v.xy();
        let m2 = v.dot(v);
        largest_velocity = if m2 > largest_velocity.z {
            v.extend(m2)
        } else {
            largest_velocity
        };
    }

    unsafe {
        output_tex.write(px.truncate(), largest_velocity.xy());
    }
}

#[spirv(compute(threads(8, 8)))]
pub fn velocity_reduce_y(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(global_invocation_id)] px: UVec3,
) {
    let mut largest_velocity = Vec3::ZERO;

    for y in 0..16 {
        let v: Vec4 = input_tex.fetch(px.xy() * uvec2(1, 16) + uvec2(0, y));
        let v = v.xy();
        let m2 = v.dot(v);
        largest_velocity = if m2 > largest_velocity.z {
            v.extend(m2)
        } else {
            largest_velocity
        };
    }

    unsafe {
        output_tex.write(px.truncate(), largest_velocity.xy());
    }
}

#[spirv(compute(threads(8, 8)))]
pub fn velocity_dilate(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(global_invocation_id)] px: IVec3,
) {
    let mut largest_velocity = Vec3::ZERO;
    let dilate_amount = 2i32;

    for x in -dilate_amount..=dilate_amount {
        for y in -dilate_amount..=dilate_amount {
            let v: Vec4 = input_tex.fetch(px.xy() + IVec2::new(x, y));
            let v = v.xy();
            let m2 = v.dot(v);
            largest_velocity = if m2 > largest_velocity.z {
                v.extend(m2)
            } else {
                largest_velocity
            };
        }
    }

    unsafe {
        output_tex.write(px.truncate(), largest_velocity.xy());
    }
}
