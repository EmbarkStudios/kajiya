use spirv_std::{glam::{UVec3, Vec2, Vec4}, Image, Sampler};

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[repr(C)]
pub struct Constants {
    output_extent_x: u32,
    output_extent_y: u32,
    self_weight: f32,
}

fn lerp(from: Vec4, to: Vec4, t: f32) -> Vec4 {
    from * (1.0 - t) + to * t
}

#[spirv(compute(threads(8, 8)))]
pub fn rev_blur_cs(
    #[spirv(descriptor_set = 0, binding = 0)] input_tail_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(descriptor_set = 0, binding = 32)] sampler_lnc: &Sampler,
    #[spirv(uniform, descriptor_set = 0, binding = 3)] constants: &Constants,
    #[spirv(global_invocation_id)] px: UVec3,
) {
    let pyramid_col: Vec4 = input_tail_tex.fetch(px.truncate());

    let mut self_col: Vec4;
    let inv_size = Vec2::ONE
        / Vec2::new(
            constants.output_extent_x as f32,
            constants.output_extent_y as f32,
        );
    if true {
        // TODO: do a small Gaussian blur instead of this nonsense

        const K: i32 = 1;
        self_col = Vec4::ZERO;

        for y in -K..=K {
            for x in -K..=K {
                let uv =
                    (px.truncate().as_f32() + Vec2::splat(0.5) + Vec2::new(x as f32, y as f32))
                        * inv_size;
                let sample: Vec4 = input_tex.sample_by_lod(*sampler_lnc, uv, 0.0);
                self_col += sample;
            }
        }

        self_col /= ((2 * K + 1) * (2 * K + 1)) as f32;
    } else {
        let uv = (px.truncate().as_f32() + Vec2::splat(0.5)) * inv_size;
        //float4 self_col = input_tex[px / 2];
        self_col = input_tex.sample_by_lod(*sampler_lnc, uv, 0.0);
    }

    let exponential_falloff = 0.5;

    // BUG: when `self_weight` is 1.0, the `w` here should be 1.0, not `exponential_falloff`
    unsafe {
        output_tex.write(
            px.truncate(),
            lerp(
                self_col, pyramid_col,
                constants.self_weight * exponential_falloff,
            ),
        );
    }
}
