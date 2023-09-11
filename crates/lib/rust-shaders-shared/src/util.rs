use crate::frame_constants::FrameConstants;
use glam::{Mat3, UVec2, UVec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use macaw::FloatExt;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub fn get_uv_u(pix: UVec2, tex_size: Vec4) -> Vec2 {
    (pix.as_vec2() + Vec2::splat(0.5)) * tex_size.zw()
}

// Replacement for abs due to SPIR-V codegen bug. See https://github.com/EmbarkStudios/rust-gpu/issues/468
pub fn abs_f32(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        -x
    }
}

// For element `i` of `self`, return `v[i].abs()`
// Work around for https://github.com/EmbarkStudios/rust-gpu/issues/468.
pub fn abs_vec2(v: Vec2) -> Vec2 {
    Vec2::new(abs_f32(v.x), abs_f32(v.y))
}

// For element `i` of `self`, return `v[i].abs()`
// Work around for https://github.com/EmbarkStudios/rust-gpu/issues/468.
pub fn abs_vec3(v: Vec3) -> Vec3 {
    Vec3::new(abs_f32(v.x), abs_f32(v.y), abs_f32(v.z))
}

// For element `i` of `self`, return `v[i].abs()`
// Work around for https://github.com/EmbarkStudios/rust-gpu/issues/468.
pub fn abs_vec4(v: Vec4) -> Vec4 {
    Vec4::new(abs_f32(v.x), abs_f32(v.y), abs_f32(v.z), abs_f32(v.w))
}

pub fn fast_sqrt(x: f32) -> f32 {
    f32::from_bits(0x1fbd1df5 + (x.to_bits() >> 1))
}

pub fn fast_sqrt_vec3(v: Vec3) -> Vec3 {
    Vec3::new(fast_sqrt(v.x), fast_sqrt(v.y), fast_sqrt(v.z))
}

// From Eberly 2014
pub fn fast_acos(x: f32) -> f32 {
    let abs_x = abs_f32(x);
    let mut res = -0.156583 * abs_x + core::f32::consts::FRAC_PI_2;
    res *= fast_sqrt(1.0 - abs_x);
    if x >= 0.0 {
        res
    } else {
        core::f32::consts::PI - res
    }
}

// Replacement for signum due to SPIR-V codegen bug. See https://github.com/EmbarkStudios/rust-gpu/issues/468
pub fn signum_f32(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

pub fn depth_to_view_z(depth: f32, frame_constants: &FrameConstants) -> f32 {
    (depth
        * -frame_constants
            .view_constants
            .clip_to_view
            .to_cols_array_2d()[2][3])
        .recip()
}

pub fn depth_to_view_z_vec4(depth: Vec4, frame_constants: &FrameConstants) -> Vec4 {
    (depth
        * -frame_constants
            .view_constants
            .clip_to_view
            .to_cols_array_2d()[2][3])
        .recip()
}

// Note: `const_mat3` is initialized with columns, while `float3x3` in HLSL is row-order,
// therefore the initializers _appear_ transposed compared to HLSL.
// The difference is only in the `top` and `bottom` ones; the others are symmetric.
pub const CUBE_MAP_FACE_ROTATIONS: [Mat3; 6] = [
    Mat3::from_cols_array_2d(&[[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]), // right
    Mat3::from_cols_array_2d(&[[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),   // left
    Mat3::from_cols_array_2d(&[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),   // top
    Mat3::from_cols_array_2d(&[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),   // bottom
    Mat3::from_cols_array_2d(&[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),  // back
    Mat3::from_cols_array_2d(&[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),  // front
];

pub fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.328_306_4e-10 // / 0x100000000
}

pub fn hammersley(i: u32, n: u32) -> Vec2 {
    Vec2::new((i + 1) as f32 / n as f32, radical_inverse_vdc(i + 1))
}

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
pub fn build_orthonormal_basis(n: Vec3) -> Mat3 {
    let b1;
    let b2;

    if n.z < 0.0 {
        let a = 1.0 / (1.0 - n.z);
        let b = n.x * n.y * a;
        b1 = Vec3::new(1.0 - n.x * n.x * a, -b, n.x);
        b2 = Vec3::new(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        b1 = Vec3::new(1.0 - n.x * n.x * a, b, -n.x);
        b2 = Vec3::new(b, 1.0 - n.y * n.y * a, -n.y);
    }

    Mat3::from_cols(b1, b2, n)
}

pub fn uniform_sample_cone(urand: Vec2, cos_theta_max: f32) -> Vec3 {
    let cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
    let sin_theta = (1.0 - cos_theta * cos_theta).saturate().sqrt();
    let phi = urand.y * core::f32::consts::TAU;
    Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

pub fn uv_to_cs(uv: Vec2) -> Vec2 {
    (uv - Vec2::new(0.5, 0.5)) * Vec2::new(2.0, -2.0)
}

pub fn cs_to_uv(cs: Vec2) -> Vec2 {
    cs * Vec2::new(0.5, -0.5) + Vec2::new(0.5, 0.5)
}

fn pack_unorm(val: f32, bit_count: u32) -> u32 {
    let max_val = (1u32 << bit_count) - 1;
    (val.clamp(0.0, 1.0) * max_val as f32) as u32
}

fn unpack_unorm(pckd: u32, bit_count: u32) -> f32 {
    let max_val = (1u32 << bit_count) - 1;
    (pckd & max_val) as f32 / max_val as f32
}

pub fn unpack_normal_11_10_11(pckd: f32) -> Vec3 {
    let p = pckd.to_bits();
    (Vec3::new(
        unpack_unorm(p, 11),
        unpack_unorm(p >> 11, 10),
        unpack_unorm(p >> 21, 11),
    ) * 2.0
        - Vec3::ONE)
        .normalize()
}

pub fn pack_normal_11_10_11(n: Vec3) -> f32 {
    let mut pckd = 0;
    pckd += pack_unorm(n.x * 0.5 + 0.5, 11);
    pckd += pack_unorm(n.y * 0.5 + 0.5, 10) << 11;
    pckd += pack_unorm(n.z * 0.5 + 0.5, 11) << 21;
    f32::from_bits(pckd)
}

pub fn pack_color_888(color: Vec3) -> u32 {
    let color = fast_sqrt_vec3(color);
    let mut pckd = 0;
    pckd += pack_unorm(color.x, 8);
    pckd += pack_unorm(color.y, 8) << 8;
    pckd += pack_unorm(color.z, 8) << 16;
    pckd
}

pub fn unpack_color_888(p: u32) -> Vec3 {
    let color = Vec3::new(
        unpack_unorm(p, 8),
        unpack_unorm(p >> 8, 8),
        unpack_unorm(p >> 16, 8),
    );
    color * color
}

pub fn unpack_unit_direction_11_10_11(pck: u32) -> Vec3 {
    Vec3::new(
        (pck & ((1u32 << 11u32) - 1u32)) as f32 * (2.0 / ((1u32 << 11u32) - 1u32) as f32) - 1.0,
        ((pck >> 11u32) & ((1u32 << 10) - 1u32)) as f32 * (2.0 / ((1u32 << 10u32) - 1u32) as f32)
            - 1.0,
        (pck >> 21) as f32 * (2.0 / ((1u32 << 11u32) - 1u32) as f32) - 1.0,
    )
}

pub fn pack_unit_direction_11_10_11(x: f32, y: f32, z: f32) -> u32 {
    let x = (x.max(-1.0).min(1.0).mul_add(0.5, 0.5) * ((1u32 << 11u32) - 1u32) as f32) as u32;
    let y = (y.max(-1.0).min(1.0).mul_add(0.5, 0.5) * ((1u32 << 10u32) - 1u32) as f32) as u32;
    let z = (z.max(-1.0).min(1.0).mul_add(0.5, 0.5) * ((1u32 << 11u32) - 1u32) as f32) as u32;

    (z << 21) | (y << 11) | x
}

// The below functions provide a simulation of ByteAddressBuffer and VertexPacked.

pub fn load2f(data: &[u32], byte_offset: u32) -> Vec2 {
    let offset = (byte_offset >> 2) as usize;
    let a = f32::from_bits(data[offset]);
    let b = f32::from_bits(data[offset + 1]);
    Vec2::new(a, b)
}

pub fn load3f(data: &[u32], byte_offset: u32) -> Vec3 {
    let offset = (byte_offset >> 2) as usize;
    let a = f32::from_bits(data[offset]);
    let b = f32::from_bits(data[offset + 1]);
    let c = f32::from_bits(data[offset + 2]);
    Vec3::new(a, b, c)
}

pub fn load4f(data: &[u32], byte_offset: u32) -> Vec4 {
    let offset = (byte_offset >> 2) as usize;
    let a = f32::from_bits(data[offset]);
    let b = f32::from_bits(data[offset + 1]);
    let c = f32::from_bits(data[offset + 2]);
    let d = f32::from_bits(data[offset + 3]);
    Vec4::new(a, b, c, d)
}

/// Decode mesh vertex from Kajiya ("core", position + normal packed together)
/// The returned normal is not normalized (but close).
pub fn load_vertex(data: &[u32], byte_offset: u32) -> (Vec3, Vec3) {
    let core_offset = (byte_offset >> 2) as usize;
    let in_pos = Vec3::new(
        f32::from_bits(data[core_offset]),
        f32::from_bits(data[core_offset + 1]),
        f32::from_bits(data[core_offset + 2]),
    );
    let in_normal = unpack_unit_direction_11_10_11(data[core_offset + 3]);
    (in_pos, in_normal)
}

pub fn store_vertex(data: &mut [u32], byte_offset: u32, position: Vec3, normal: Vec3) {
    let offset = (byte_offset >> 2) as usize;
    let packed_normal = pack_unit_direction_11_10_11(normal.x, normal.y, normal.z);
    data[offset] = position.x.to_bits();
    data[offset + 1] = position.y.to_bits();
    data[offset + 2] = position.z.to_bits();
    data[offset + 3] = packed_normal;
}

pub fn unpack_u32_to_vec4(v: u32) -> Vec4 {
    Vec4::new(
        (v & 0xFF) as f32 / 255.0,
        ((v >> 8) & 0xFF) as f32 / 255.0,
        ((v >> 16) & 0xFF) as f32 / 255.0,
        ((v >> 24) & 0xFF) as f32 / 255.0,
    )
}

pub fn roughness_to_perceptual_roughness(r: f32) -> f32 {
    r.sqrt()
}

pub fn perceptual_roughness_to_roughness(r: f32) -> f32 {
    r * r
}

const RGB9E5_EXPONENT_BITS: u32 = 5;
const RGB9E5_MANTISSA_BITS: u32 = 9;
const RGB9E5_EXP_BIAS: u32 = 15;
const RGB9E5_MAX_VALID_BIASED_EXP: u32 = 31;
const MAX_RGB9E5_EXP: u32 = RGB9E5_MAX_VALID_BIASED_EXP - RGB9E5_EXP_BIAS;
const RGB9E5_MANTISSA_VALUES: u32 = 1 << RGB9E5_MANTISSA_BITS;
const MAX_RGB9E5_MANTISSA: u32 = RGB9E5_MANTISSA_VALUES - 1;
const MAX_RGB9E5: f32 =
    (MAX_RGB9E5_MANTISSA as f32 / RGB9E5_MANTISSA_VALUES as f32) * (1 << MAX_RGB9E5_EXP) as f32;

fn clamp_range_for_rgb9e5(x: f32) -> f32 {
    x.clamp(0.0, MAX_RGB9E5)
}

fn floor_log2_positive(x: f32) -> i32 {
    // float bit hacking. Wonder if .log2().floor() wouldn't be faster.
    let f = x.to_bits();
    (f >> 23) as i32 - 127
}

// workaround rust-gpu bug, will be fixed by #690
fn mymax(a: i32, b: i32) -> i32 {
    if a >= b {
        a
    } else {
        b
    }
}

// https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_shared_exponent.txt
pub fn float3_to_rgb9e5(rgb: Vec3) -> u32 {
    let rc = clamp_range_for_rgb9e5(rgb.x);
    let gc = clamp_range_for_rgb9e5(rgb.y);
    let bc = clamp_range_for_rgb9e5(rgb.z);

    let maxrgb = rc.max(gc.max(bc));
    let mut exp_shared: i32 = mymax(-(RGB9E5_EXP_BIAS as i32) - 1, floor_log2_positive(maxrgb))
        + 1
        + RGB9E5_EXP_BIAS as i32;
    let mut denom =
        ((exp_shared - RGB9E5_EXP_BIAS as i32 - RGB9E5_MANTISSA_BITS as i32) as f32).exp2();

    let maxm = (maxrgb / denom + 0.5).floor() as i32;
    if maxm == (MAX_RGB9E5_MANTISSA as i32 + 1) {
        denom *= 2.0;
        exp_shared += 1;
    }

    let rm = (rc / denom + 0.5).floor() as u32;
    let gm = (gc / denom + 0.5).floor() as u32;
    let bm = (bc / denom + 0.5).floor() as u32;

    (rm << (32 - 9)) | (gm << (32 - 9 * 2)) | (bm << (32 - 9 * 3)) | (exp_shared as u32)
}

pub fn rgb9e5_to_float3(v: u32) -> Vec3 {
    let exponent = bitfield_extract(v, 0, RGB9E5_EXPONENT_BITS) as i32
        - RGB9E5_EXP_BIAS as i32
        - RGB9E5_MANTISSA_BITS as i32;
    let scale = (exponent as f32).exp2();

    Vec3::new(
        bitfield_extract(v, 32 - RGB9E5_MANTISSA_BITS, RGB9E5_MANTISSA_BITS) as f32 * scale,
        bitfield_extract(v, 32 - RGB9E5_MANTISSA_BITS * 2, RGB9E5_MANTISSA_BITS) as f32 * scale,
        bitfield_extract(v, 32 - RGB9E5_MANTISSA_BITS * 3, RGB9E5_MANTISSA_BITS) as f32 * scale,
    )
}

fn bitfield_extract(value: u32, offset: u32, bits: u32) -> u32 {
    let mask = (1 << bits) - 1;
    (value >> offset) & mask
}

pub fn hash1(mut x: u32) -> u32 {
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;
    x
}

pub fn hash_combine2(x: u32, y: u32) -> u32 {
    const M: u32 = 1664525;
    const C: u32 = 1013904223;
    let mut seed = (x * M + y + C) * M;
    // Tempering (from Matsumoto)
    seed ^= seed >> 11;
    seed ^= (seed << 7) & 0x9d2c5680;
    seed ^= (seed << 15) & 0xefc60000;
    seed ^= seed >> 18;
    seed
}

pub fn hash2(v: UVec2) -> u32 {
    hash_combine2(v.x, hash1(v.y))
}

pub fn hash3(v: UVec3) -> u32 {
    hash_combine2(v.x, hash2(v.yz()))
}

pub fn uint_to_u01_float(h: u32) -> f32 {
    const MANTISSA_MASK: u32 = 0x007FFFFF;
    const ONE: u32 = 0x3F800000;
    f32::from_bits((h & MANTISSA_MASK) | ONE) - 1.0
}

pub fn sign(val: f32) -> f32 {
    ((0.0 < val) as i32 - (val < 0.0) as i32) as f32
}
