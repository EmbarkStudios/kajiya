use crate::frame_constants::FrameConstants;
use macaw::{const_mat3, FloatExt, Mat3, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub fn get_uv_u(pix: UVec2, tex_size: Vec4) -> Vec2 {
    (pix.as_vec2() + Vec2::splat(0.5)) * tex_size.zw()
}

pub fn depth_to_view_z(depth: f32, frame_constants: &FrameConstants) -> f32 {
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
    const_mat3!([0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]), // right
    const_mat3!([0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]),   // left
    const_mat3!([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),   // top
    const_mat3!([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),   // bottom
    const_mat3!([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),  // back
    const_mat3!([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),  // front
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
