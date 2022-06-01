// not used

use crate::pack_unpack::unpack_normal_11_10_11_no_normalize;
use macaw::{IVec2, UVec3, Vec3, Vec4};
use rust_shaders_shared::frame_constants::FrameConstants;
use spirv_std::Image;

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[spirv(compute(threads(8, 8)))]
pub fn extract_half_res_gbuffer_view_normal_rgba8(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(uniform, descriptor_set = 2, binding = 0)] frame_constants: &FrameConstants,

    #[spirv(global_invocation_id)] id: UVec3,
) {
    let px = id.truncate();

    let hi_px_subpixels: [IVec2; 4] = [
        IVec2::new(0, 0),
        IVec2::new(1, 1),
        IVec2::new(1, 0),
        IVec2::new(0, 1),
    ];

    let src_px: IVec2 =
        px.as_ivec2() * 2 + hi_px_subpixels[(frame_constants.frame_index & 3) as usize];

    // TODO: use gbuffer unpacking
    let input: Vec4 = input_tex.fetch(IVec2::new(src_px.x as i32, src_px.y as i32));
    let normal: Vec3 = unpack_normal_11_10_11_no_normalize(input.y);
    let normal_vs: Vec3 = frame_constants
        .view_constants
        .world_to_view
        .transform_vector3(normal)
        .normalize();
    unsafe {
        output_tex.write(id.truncate(), normal_vs.extend(1.0));
    }
}
