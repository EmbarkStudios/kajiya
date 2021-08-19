#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, asm),
    register_attr(spirv)
)]

#[cfg(not(target_arch = "spirv"))]
#[macro_use]
pub extern crate spirv_std_macros;

pub mod frame_constants;
pub mod motion_blur;
pub mod util;
