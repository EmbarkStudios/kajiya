// Lots of Tomasz' prototyping remainders in this file, plus disabled dither for now.
#![allow(dead_code)]

use crate::util::{abs_f32, get_uv_u, signum_f32};
use crate::frame_constants::FrameConstants;
use crate::tonemap::*;
use crate::color::lin_srgb_to_luminance;
use macaw::{lerp, IVec2, UVec3, Vec3, Vec4};
use spirv_std::{Image, Sampler};

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Constants {
    output_tex_size: Vec4,
}

const USE_TONEMAP: bool = true;
const USE_DITHER: bool = true;
const USE_SHARPEN: bool = true;

const GLARE_AMOUNT: f32 = 0.07;

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

fn local_tmo_constrain(x: f32, max_compression: f32) -> f32 {
    const LOCAL_TMO_CONSTRAIN_MODE: usize = 2;

    match LOCAL_TMO_CONSTRAIN_MODE {
        0 => ((x.ln() / max_compression).tanh() * max_compression).exp(),
        1 => {
            let mut x = x.ln();
            let s = signum_f32(x);
            x = abs_f32(x).sqrt();
            x = (x / max_compression).tanh() * max_compression;
            x = (x * x * s).exp();
            x
        }
        2 => {
            let k = 3.0 * max_compression;
            let mut x = 1.0 / x;
            x = tonemap_curve(x / k) * k;
            x = 1.0 / x;
            x = tonemap_curve(x / k) * k;
            x
        }
        _ => x,
    }
}

fn bitwise_and(a: IVec2, b: IVec2) -> IVec2 {
    IVec2::new(a.x & b.x, a.y & b.y)
}

#[spirv(compute(threads(8, 8)))]
pub fn post_combine_cs(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] rev_blur_pyramid_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] blue_noise_lut: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] output_tex: &Image!(2D, type=f32, sampled=false),

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
        let sharpen_amount = 0.3;

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
            let wt = wt.min(sharpen_amount * wt * 1.25);

            neighbors += n0 * wt;
            neighbors += n1 * wt;
            wt_sum += wt * 2.0;
        }

        let mut sharpened_luma = (center * (wt_sum + 1.0) - neighbors).max(0.0);
        sharpened_luma = sharpen_inv_remap(sharpened_luma);

        col *= (sharpened_luma / lin_srgb_to_luminance(col).max(1e-5)).max(0.0);
    }

    col = lerp(col..=glare, GLARE_AMOUNT);

    if USE_TONEMAP {
        col = neutral_tonemap(col);
    }

    if USE_DITHER {
        // Dither
        let urand_idx = frame_constants.frame_index as i32;
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
