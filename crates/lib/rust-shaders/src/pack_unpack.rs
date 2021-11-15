#![allow(dead_code)]

use macaw::Vec3;

fn unpack_unorm(pckd: u32, bit_count: u32) -> f32 {
    let max_val = (1u32 << bit_count) - 1;
    (pckd & max_val) as f32 / max_val as f32
}

fn pack_unorm(val: f32, bit_count: u32) -> u32 {
    let max_val = (1u32 << bit_count) - 1;
    (val.clamp(0.0, 1.0) * max_val as f32) as u32
}

pub fn unpack_normal_11_10_11_no_normalize(pckd: f32) -> Vec3 {
    let p = pckd.to_bits();
    Vec3::new(
        unpack_unorm(p, 11),
        unpack_unorm(p >> 11, 10),
        unpack_unorm(p >> 21, 11),
    ) * 2.0
        - Vec3::ONE
}
