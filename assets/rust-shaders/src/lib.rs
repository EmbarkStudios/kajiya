#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, asm),
    register_attr(spirv)
)]
#![allow(clippy::too_many_arguments)]

pub mod atmosphere;
pub mod convolve_cube;
pub mod extract_half_res_depth;
pub mod extract_half_res_gbuffer_view_normal_rgba8;
pub mod frame_constants;
pub mod motion_blur;
pub mod pack_unpack;
pub mod rev_blur;
pub mod sky;
pub mod util;
