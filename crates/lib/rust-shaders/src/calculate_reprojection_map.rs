// not used

#![allow(dead_code)]

use crate::bilinear::Bilinear;
use macaw::{vec4, IVec2, UVec3, Vec2, Vec2Ext, Vec3, Vec4, Vec4Ext, Vec4Swizzles};
use rust_shaders_shared::{
    frame_constants::FrameConstants,
    util::{abs_vec2, abs_vec4, cs_to_uv, depth_to_view_z_vec4, get_uv_u, uv_to_cs},
};
use spirv_std::{Image, Sampler};

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

// From "Fast Denoising with Self Stabilizing Recurrent Blurs"

#[repr(C)]
pub struct Constants {
    output_tex_size: Vec4,
}

fn all_above_equal(v: IVec2, value: i32) -> bool {
    v.x >= value && v.y >= value
}

fn all_below(v: IVec2, value: IVec2) -> bool {
    v.x < value.x && v.y < value.y
}

#[rustfmt::skip] // eats long spirv lines!
#[spirv(compute(threads(8, 8, 1)))]
pub fn calculate_reprojection_map_cs(
    #[spirv(descriptor_set = 0, binding = 0)] depth_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] geometric_normal_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] prev_depth_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] velocity_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] output_tex: &Image!(2D, format=rgba16f, sampled=false, arrayed=true),
    #[spirv(descriptor_set = 0, binding = 32)] sampler_lnc: &Sampler,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] constants: &Constants,

    #[spirv(uniform, descriptor_set = 2, binding = 0)] frame_constants: &FrameConstants,
    #[spirv(global_invocation_id)] px: UVec3,
) {
    let uv = get_uv_u(px.truncate(), constants.output_tex_size);

    let depth: Vec4 = depth_tex.fetch(px.truncate());
    let depth = depth.x;

    if depth == 0.0 {
        let pos_cs = uv_to_cs(uv).extend(0.0).extend(1.0);
        let pos_vs = frame_constants.view_constants.clip_to_view * pos_cs;

        let prev_vs = pos_vs;

        let prev_cs = frame_constants.view_constants.view_to_clip * prev_vs;
        let prev_pcs = frame_constants.view_constants.clip_to_prev_clip * prev_cs;

        let prev_uv = cs_to_uv(prev_pcs.xy());
        let uv_diff = prev_uv - uv;

        unsafe {
            output_tex.write(px, uv_diff.extend(0.0).extend(0.0));
        }
        return;
    }

    let geom_texel: Vec4 = geometric_normal_tex.fetch(px.truncate());
    let normal_vs: Vec3 = geom_texel.truncate() * 2.0 - 1.0;

    let pos_cs: Vec4 = uv_to_cs(uv).extend(depth).extend(1.0);
    let pos_vs: Vec4 = frame_constants.view_constants.clip_to_view * pos_cs;
    let dist_to_point = -(pos_vs.z / pos_vs.w);

    let velocity_texel: Vec4 = velocity_tex.fetch(px.truncate());
    let prev_vs: Vec4 = pos_vs / pos_vs.w + velocity_texel.truncate().extend(0.0);

    //float4 prev_cs = mul(frame_constants.view_constants.prev_view_to_prev_clip, prev_vs);
    let prev_cs: Vec4 = frame_constants.view_constants.view_to_clip * prev_vs;
    let prev_pcs: Vec4 = frame_constants.view_constants.clip_to_prev_clip * prev_cs;

    let mut prev_uv = cs_to_uv(prev_pcs.xy() / prev_pcs.w);
    let mut uv_diff: Vec2 = prev_uv - uv;

    // Account for quantization of the `uv_diff` in R16G16B16A16_SNORM.
    // This is so we calculate validity masks for pixels that the users will actually be using.
    uv_diff = (uv_diff * 32767.0 + Vec2::splat(0.5)).trunc() / 32767.0;
    prev_uv = uv + uv_diff;

    let mut prev_pvs: Vec4 = frame_constants.view_constants.prev_clip_to_prev_view * prev_pcs;
    prev_pvs /= prev_pvs.w;

    // Based on "Fast Denoising with Self Stabilizing Recurrent Blurs"

    // Note: departure from the quoted technique: they calculate reprojected sample depth by linearly
    // scaling plane distance with view-space Z, which is not correct unless the plane is aligned with view.
    // Instead, the amount that distance actually increases with depth is simply `normal_vs.z`.

    // Note: bias the minimum distance increase, so that reprojection at grazing angles has a sharper cutoff.
    // This can introduce shimmering a grazing angles, but also reduces reprojection artifacts on surfaces
    // which flip their normal from back- to fron-facing across a frame. Such reprojection smears a few
    // pixels along a wide area, creating a glitchy look.
    let plane_dist_prev_dz = normal_vs.z.min(-0.2);
    //float plane_dist_prev_dz = -normal_vs.z;

    let bilinear_at_prev = Bilinear::new(prev_uv, constants.output_tex_size.xy());
    let prev_gather_uv: Vec2 = (bilinear_at_prev.origin + Vec2::ONE) / constants.output_tex_size.xy();

    let prev_depth: Vec4 = prev_depth_tex.gather(*sampler_lnc, prev_gather_uv, 0);
    let prev_depth = prev_depth.wzxy();

    let prev_view_z: Vec4 = depth_to_view_z_vec4(prev_depth, frame_constants);

    // Note: departure from the quoted technique: linear offset from zero distance at previous position instead of scaling.
    let quad_dists: Vec4 = abs_vec4(plane_dist_prev_dz * (prev_view_z - prev_pvs.z));

    // TODO: reject based on normal too? Potentially tricky under rotations.

    let acceptance_threshold: f32 = 0.001 * (1080.0 / constants.output_tex_size.y);

    // Reduce strictness at grazing angles, where distances grow due to perspective
    let pos_vs_norm: Vec3 = (pos_vs.truncate() / pos_vs.w).normalize();
    let ndotv: f32 = normal_vs.dot(pos_vs_norm);

    let mut quad_validity: Vec4 = quad_dists.step(Vec4::splat(acceptance_threshold * dist_to_point / -ndotv));

    let out_tex_size = constants.output_tex_size.xy().as_ivec2();
    quad_validity.x *= (all_above_equal(bilinear_at_prev.px0(), 0)
        && all_below(bilinear_at_prev.px0(), out_tex_size)) as i32 as f32;
    quad_validity.y *= (all_above_equal(bilinear_at_prev.px1(), 0)
        && all_below(bilinear_at_prev.px1(), out_tex_size)) as i32 as f32;
    quad_validity.z *= (all_above_equal(bilinear_at_prev.px2(), 0)
        && all_below(bilinear_at_prev.px2(), out_tex_size)) as i32 as f32;
    quad_validity.w *= (all_above_equal(bilinear_at_prev.px3(), 0)
        && all_below(bilinear_at_prev.px3(), out_tex_size)) as i32 as f32;

    let validity = quad_validity.dot(vec4(1.0, 2.0, 4.0, 8.0)) / 15.0;

    let texel_center_offset: Vec2 = abs_vec2(Vec2::splat(0.5) - (prev_uv * constants.output_tex_size.xy()).fract());
    let accuracy = 1.0 - texel_center_offset.x - texel_center_offset.y;

    unsafe {
        output_tex.write(px, vec4(uv_diff.x, uv_diff.y, validity, accuracy));
    }
}
