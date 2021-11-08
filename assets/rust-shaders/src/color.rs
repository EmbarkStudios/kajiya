use macaw::{const_mat3, Mat3, Vec3};

pub fn lin_srgb_to_ycbcr(col: Vec3) -> Vec3 {
    // NOTE! This matrix needs to be transposed from the HLSL equivalent.
    const M: Mat3 =
        const_mat3!([0.2126, -0.1146, 0.5, 0.7152, -0.3854, -0.4542, 0.0722, 0.5, -0.0458]);
    M * col
}

pub fn ycbcr_to_lin_srgb(col: Vec3) -> Vec3 {
    // NOTE! This matrix needs to be transposed from the HLSL equivalent.
    const M: Mat3 = const_mat3!([1.0, 1.0, 1.0, 0.0, -0.1873, 1.8556, 1.5748, -0.4681, 0.0]);
    M * col
}

/// Convert linear sRGB color to monochrome luminance value
pub fn lin_srgb_to_luminance(color_lin_srgb: Vec3) -> f32 {
    Vec3::new(0.2126, 0.7152, 0.0722).dot(color_lin_srgb)
}