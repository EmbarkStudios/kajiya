use macaw::{IVec2, UVec2, UVec3, Vec4};
use spirv_std::{
    arch::control_barrier,
    memory::{Scope, Semantics},
    Image,
};

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

const KERNEL_RADIUS: i32 = 5;
const GROUP_WIDTH: u32 = 64;
const VBLUR_WINDOW_SIZE: usize = ((GROUP_WIDTH + KERNEL_RADIUS as u32) * 2) as usize;

fn gaussian_wt(dst_px: f32, src_px: f32) -> f32 {
    let px_off = (dst_px + 0.5) * 2.0 - (src_px + 0.5);
    let sigma = KERNEL_RADIUS as f32 * 0.5;
    (-px_off * px_off / (sigma * sigma)).exp()
}

fn vblur(input_tex: &Image!(2D, type=f32, sampled=true), dst_px: IVec2, src_px: IVec2) -> Vec4 {
    let mut res = Vec4::ZERO;
    let mut wt_sum = 0.0f32;

    let mut y = 0;
    while y < KERNEL_RADIUS * 2 {
        let wt = gaussian_wt(dst_px.y as f32, (src_px.y + y) as f32);
        let value: Vec4 = input_tex.fetch(src_px + IVec2::new(0, y));
        res += value * wt;
        wt_sum += wt;
        y += 1;
    }

    res / wt_sum
}

fn vblur_into_shmem(
    input_tex: &Image!(2D, type=f32, sampled=true),
    vblur_out: &mut [Vec4; VBLUR_WINDOW_SIZE],
    dst_px: IVec2,
    xfetch: i32,
    group_id: UVec2,
) {
    let src_px: IVec2 = group_id.as_ivec2() * IVec2::new(GROUP_WIDTH as i32 * 2, 2)
        + IVec2::new(xfetch - KERNEL_RADIUS, -KERNEL_RADIUS);
    vblur_out[xfetch as usize] = vblur(input_tex, dst_px, src_px);
}

#[spirv(compute(threads(64, 1, 1)))] // 64 == GROUP_WIDTH
pub fn blur_cs(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(workgroup)] vblur_out: &mut [Vec4; VBLUR_WINDOW_SIZE], // groupshared float4 vblur_out[VBLUR_WINDOW_SIZE];
    #[spirv(global_invocation_id)] px: UVec3,
    #[spirv(local_invocation_id)] px_within_group: UVec3,
    #[spirv(workgroup_id)] group_id: UVec3,
) {
    let px = px.truncate();
    let group_id = group_id.truncate();
    let mut xfetch = px_within_group.x;
    while xfetch < VBLUR_WINDOW_SIZE as u32 {
        vblur_into_shmem(input_tex, vblur_out, px.as_ivec2(), xfetch as i32, group_id);
        xfetch += GROUP_WIDTH;
    }

    // GroupMemoryBarrierWithGroupSync();
    unsafe {
        control_barrier::<
            { Scope::Workgroup as u32 },
            { Scope::Workgroup as u32 },
            { Semantics::WORKGROUP_MEMORY.bits() | Semantics::ACQUIRE_RELEASE.bits() },
        >();
    }

    let mut res = Vec4::ZERO;
    let mut wt_sum = 0.0;

    for x in 0..=(KERNEL_RADIUS * 2) {
        let wt = gaussian_wt(px.x as f32, (px.x as i32 * 2 + x - KERNEL_RADIUS) as f32);
        res += vblur_out[(px_within_group.x * 2 + x as u32) as usize] * wt;
        wt_sum += wt;
    }
    res /= wt_sum;

    unsafe {
        output_tex.write(px, res);
    }
}
