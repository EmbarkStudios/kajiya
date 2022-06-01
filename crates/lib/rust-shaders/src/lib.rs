#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, asm),
    register_attr(spirv)
)]
#![allow(clippy::too_many_arguments)]

pub mod bilinear;
pub mod blur;
pub mod color;
pub mod constants;
pub mod copy_depth_to_r;
pub mod gbuffer;
pub mod motion_blur;
pub mod pack_unpack;
pub mod rev_blur;
pub mod ssgi;
