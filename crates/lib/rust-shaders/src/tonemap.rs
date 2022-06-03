// not used

use crate::color;
use macaw::{prelude::*, Vec3};

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub fn tonemap_curve(v: f32) -> f32 {
    /*
    // Large linear part in the lows, but compresses highs.
    let c = v + v * v + 0.5 * v * v * v;
    c / (1.0 + c)
    */
    1.0 - (-v).exp()
}

pub fn tonemap_curve_vec3(v: Vec3) -> Vec3 {
    Vec3::new(tonemap_curve(v.x), tonemap_curve(v.y), tonemap_curve(v.z))

    // Vector operations for smaller SPIR-V.
    /*let v2 = v * v;
    let c = v + v2 + 0.5 * v2 * v;
    c / (Vec3::ONE + c)*/
}

pub fn neutral_tonemap(col: Vec3) -> Vec3 {
    let ycbcr = color::lin_srgb_to_ycbcr(col);

    let bt = tonemap_curve(ycbcr.yz().length() * 2.4);
    let desat = ((bt - 0.7) * 0.8).max(0.0);
    let desat = desat * desat;

    let desat_col = col.lerp(Vec3::splat(ycbcr.x), desat);

    let tm_luma = tonemap_curve(ycbcr.x);
    let tm0 = col * (tm_luma / color::lin_srgb_to_luminance(col).max(1e-5)).max(0.0);
    let final_mult = 0.97;
    let tm1 = tonemap_curve_vec3(desat_col);

    tm0.lerp(tm1, bt * bt) * final_mult
}
