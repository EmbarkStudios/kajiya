#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, asm),
    register_attr(spirv)
)]
#![allow(clippy::too_many_arguments)]

pub mod atmosphere;
pub mod bilinear;
pub mod calculate_reprojection_map;
pub mod color;
pub mod constants;
pub mod convolve_cube;
pub mod copy_depth_to_r;
pub mod frame_constants;
pub mod motion_blur;
pub mod post_combine;
pub mod rev_blur;
pub mod sky;
pub mod tonemap;
pub mod util;
