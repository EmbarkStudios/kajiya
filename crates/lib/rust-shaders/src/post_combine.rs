// Lots of prototyping remainders in this file
#![allow(dead_code)]

use crate::{color::lin_srgb_to_luminance, tonemap::*};
use macaw::{lerp, IVec2, UVec3, Vec3, Vec4};
use rust_shaders_shared::{
    frame_constants::FrameConstants,
    util::{abs_f32, get_uv_u, signum_f32},
};
use spirv_std::{Image, RuntimeArray, Sampler};

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Constants {
    output_tex_size: Vec4,
    ev_shift: f32,
}

const USE_GRADE: bool = true;
const USE_TONEMAP: bool = true;
const USE_DITHER: bool = true;
const USE_SHARPEN: bool = true;
const USE_VIGNETTE: bool = true;

const SHARPEN_AMOUNT: f32 = 0.1;
const GLARE_AMOUNT: f32 = 0.05;

fn rsqrt(f: f32) -> f32 {
    1.0 / f.sqrt()
}

fn sharpen_remap(l: f32) -> f32 {
    l.sqrt()
}

fn sharpen_inv_remap(l: f32) -> f32 {
    l * l
}

fn triangle_remap(n: f32) -> f32 {
    let origin = n * 2.0 - 1.0;
    let v = origin * rsqrt(abs_f32(origin));
    v.max(-1.0) - signum_f32(origin)
}

fn bitwise_and(a: IVec2, b: IVec2) -> IVec2 {
    IVec2::new(a.x & b.x, a.y & b.y)
}

trait Smoothstep: Sized {
    fn smoothstep(edge0: Self, edge1: Self, x: Self) -> Self;
}

impl Smoothstep for f32 {
    fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        let x = ((x - edge0) / (edge1 - edge0)).max(0.0).min(1.0);
        x * x * (3.0 - 2.0 * x)
    }
}

impl Smoothstep for Vec3 {
    fn smoothstep(edge0: Vec3, edge1: Vec3, x: Vec3) -> Vec3 {
        Vec3::new(
            <f32 as Smoothstep>::smoothstep(edge0.x, edge1.x, x.x),
            <f32 as Smoothstep>::smoothstep(edge0.y, edge1.y, x.y),
            <f32 as Smoothstep>::smoothstep(edge0.z, edge1.z, x.z),
        )
    }
}

// (Very) reduced version of:
// Uchimura 2017, "HDR theory and practice"
// Math: https://www.desmos.com/calculator/gslcdxvipg
// Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
//  m: linear section start
//  c: black
fn push_down_black_point(x: Vec3, m: f32, c: f32) -> Vec3 {
    let w0 = Vec3::ONE - Vec3::smoothstep(Vec3::ZERO, Vec3::splat(m), x);
    let w1 = 1.0 - w0;

    let t = (x / m).powf(c) * m;
    t * w0 + x * w1
}

#[spirv(compute(threads(8, 8)))]
pub fn post_combine_cs(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] rev_blur_pyramid_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(descriptor_set = 1, binding = 3)] bindless_textures: &RuntimeArray<
        Image!(2D, type=f32, sampled=true),
    >,

    #[spirv(descriptor_set = 0, binding = 32)] sampler_lnc: &Sampler,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] constants: &Constants,
    #[spirv(uniform, descriptor_set = 2, binding = 0)] frame_constants: &FrameConstants,
    #[spirv(global_invocation_id)] px: UVec3,
) {
    let uv = get_uv_u(px.truncate(), constants.output_tex_size);

    let glare_vec4: Vec4 = rev_blur_pyramid_tex.sample_by_lod(*sampler_lnc, uv, 0.0);
    let glare: Vec3 = glare_vec4.truncate();

    let col: Vec4 = input_tex.fetch(px.truncate());
    let mut col: Vec3 = col.truncate();

    // TODO: move to its own pass
    if USE_SHARPEN {
        let mut neighbors = 0.0;
        let mut wt_sum = 0.0;

        let dim_offsets: [IVec2; 2] = [IVec2::new(1, 0), IVec2::new(0, 1)];

        let center = sharpen_remap(lin_srgb_to_luminance(col));

        #[allow(clippy::needless_range_loop)]
        for dim in 0..2 {
            let n0coord: IVec2 = px.truncate().as_ivec2() + dim_offsets[dim];
            let n1coord: IVec2 = px.truncate().as_ivec2() - dim_offsets[dim];

            let n0_texel: Vec4 = input_tex.fetch(n0coord);
            let n1_texel: Vec4 = input_tex.fetch(n1coord);
            let n0 = sharpen_remap(lin_srgb_to_luminance(n0_texel.truncate()));
            let n1 = sharpen_remap(lin_srgb_to_luminance(n1_texel.truncate()));
            let wt = 0f32.max(1.0 - 6.0 * (abs_f32(center - n0) + abs_f32(center - n1)));
            let wt = wt.min(SHARPEN_AMOUNT * wt * 1.25);

            neighbors += n0 * wt;
            neighbors += n1 * wt;
            wt_sum += wt * 2.0;
        }

        let mut sharpened_luma = (center * (wt_sum + 1.0) - neighbors).max(0.0);
        sharpened_luma = sharpen_inv_remap(sharpened_luma);

        col *= (sharpened_luma / lin_srgb_to_luminance(col).max(1e-5)).max(0.0);
    }

    col = lerp(col..=glare, GLARE_AMOUNT);
    col = col.max(Vec3::ZERO);
    col *= constants.ev_shift.exp2();

    if USE_VIGNETTE {
        col *= (-2.0 * (uv - 0.5).length().powi(3)).exp();
    }

    if USE_GRADE {
        // Lift mids
        col = col.powf(0.9);

        // Push down lows
        col = push_down_black_point(col, 0.2, 1.25);
    }

    if USE_TONEMAP {
        col = neutral_tonemap(col);
    }

    if USE_DITHER {
        // Dither
        let urand_idx = frame_constants.frame_index as i32;
        let blue_noise_lut = unsafe {
            bindless_textures.index(crate::constants::BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0)
        };
        // 256x256 blue noise
        let dither = triangle_remap({
            let value: Vec4 = blue_noise_lut.fetch(bitwise_and(
                px.truncate().as_ivec2() + IVec2::new(urand_idx * 59, urand_idx * 37),
                IVec2::splat(255),
            ));
            value.x
        });

        col += Vec3::splat(dither / 256.0);
    }

    unsafe {
        output_tex.write(px.truncate(), col.extend(1.0));
    }
}
