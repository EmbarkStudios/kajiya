use glam::{UVec3, Vec4};
use spirv_std::Image;

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[spirv(compute(threads(8, 8)))]
pub fn copy_depth_to_r_cs(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let color: Vec4 = input_tex.fetch(id.truncate());
    unsafe {
        output_tex.write(id.truncate(), color);
    }
}
