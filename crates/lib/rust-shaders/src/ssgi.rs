use macaw::{
    ivec2, uvec3, vec2, vec4, FloatExt, IVec3, UVec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Ext,
    Vec4Swizzles,
};
use rust_shaders_shared::{
    frame_constants::FrameConstants, gbuffer::*, ssgi::SsgiConstants, util::*,
    view_ray::ViewRayContext,
};
use spirv_std::{Image, Sampler};

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

fn ssgi_kernel_radius(constants: &SsgiConstants) -> f32 {
    if constants.use_ao_only {
        constants.kernel_radius * constants.output_tex_size.w
    } else {
        constants.kernel_radius
    }
}

fn process_upsample_sample(
    soffset: Vec2,
    ssgi: Vec4,
    depth: f32,
    normal: Vec3,
    center_depth: f32,
    center_normal: Vec3,
    w_sum: &mut f32,
) -> Vec4 {
    if depth != 0.0 {
        let depth_diff = 1.0 - (center_depth / depth);
        let depth_factor = (-200.0 * depth_diff.abs()).exp2();

        let mut normal_factor = 0.0f32.max(normal.dot(center_normal));
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;

        let mut w = 1.0;
        w *= depth_factor; // TODO: differentials
        w *= normal_factor;
        w *= (-soffset.dot(soffset)).exp();

        *w_sum += w;
        ssgi * w
    } else {
        Vec4::ZERO
    }
}

fn process_spatial_filter_sample(
    ssgi: Vec4,
    depth: f32,
    normal: Vec3,
    center_depth: f32,
    center_normal: Vec3,
    w_sum: &mut f32,
) -> Vec4 {
    if depth != 0.0 {
        let depth_diff = 1.0 - (center_depth / depth);
        let depth_factor = (-200.0 * depth_diff.abs()).exp2();

        let mut normal_factor = 0.0f32.max(normal.dot(center_normal));
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;

        let mut w = 1.0;
        w *= depth_factor; // TODO: differentials
        w *= normal_factor;

        *w_sum += w;
        ssgi * w
    } else {
        Vec4::ZERO
    }
}

#[spirv(compute(threads(8, 8)))]
pub fn upsample_cs(
    #[spirv(descriptor_set = 0, binding = 0)] ssgi_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] depth_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] gbuffer_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(global_invocation_id)] px: IVec3,
) {
    let mut result;
    let mut w_sum = 0.0f32;

    let center_depth: Vec4 = depth_tex.fetch(px.xy());
    let center_depth = center_depth.x;

    if center_depth != 0.0 {
        let center_normal: Vec4 = gbuffer_tex.fetch(px.xy());
        let center_normal = unpack_normal_11_10_11(center_normal.y);

        let center_ssgi = Vec4::ZERO;
        w_sum = 0.0f32;
        result = center_ssgi;

        let kernel_half_size = 1i32;
        for y in -kernel_half_size..=kernel_half_size {
            for x in -kernel_half_size..=kernel_half_size {
                let sample_pix = px.xy() / 2 + ivec2(x, y);
                let depth: Vec4 = depth_tex.fetch(sample_pix * 2);
                let depth = depth.x;
                let ssgi: Vec4 = ssgi_tex.fetch(sample_pix);
                let normal: Vec4 = gbuffer_tex.fetch(sample_pix * 2);
                let normal = unpack_normal_11_10_11(normal.y);
                result += process_upsample_sample(
                    vec2(x as f32, y as f32),
                    ssgi,
                    depth,
                    normal,
                    center_depth,
                    center_normal,
                    &mut w_sum,
                );
            }
        }
    } else {
        result = Vec4::ZERO;
    }

    unsafe {
        if w_sum > 1e-6 {
            output_tex.write(px.truncate(), result / w_sum);
        } else {
            let result: Vec4 = ssgi_tex.fetch(px.xy() / 2);
            output_tex.write(px.truncate(), result);
        }
    }
}

#[spirv(compute(threads(8, 8)))]
pub fn spatial_filter_cs(
    #[spirv(descriptor_set = 0, binding = 0)] ssgi_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] depth_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] normal_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(global_invocation_id)] px: IVec3,
) {
    let mut result;
    let mut w_sum = 0.0f32;

    let center_depth: Vec4 = depth_tex.fetch(px.xy());
    let center_depth = center_depth.x;

    if center_depth != 0.0 {
        let center_normal: Vec4 = normal_tex.fetch(px.xy());
        let center_normal = center_normal.xyz();

        let center_ssgi: Vec4 = ssgi_tex.fetch(px.xy());
        w_sum = 1.0f32;
        result = center_ssgi;

        let kernel_half_size = 1i32;
        for y in -kernel_half_size..=kernel_half_size {
            for x in -kernel_half_size..=kernel_half_size {
                if x != 0 || y != 0 {
                    let sample_px = px.xy() + ivec2(x, y);
                    let depth: Vec4 = depth_tex.fetch(sample_px);
                    let depth = depth.x;
                    let ssgi: Vec4 = ssgi_tex.fetch(sample_px);
                    let normal: Vec4 = normal_tex.fetch(sample_px);
                    let normal = normal.xyz();
                    result += process_spatial_filter_sample(
                        ssgi,
                        depth,
                        normal,
                        center_depth,
                        center_normal,
                        &mut w_sum,
                    );
                }
            }
        }
    } else {
        result = Vec4::ZERO;
    }

    unsafe {
        output_tex.write(px.truncate(), result / w_sum.max(1e-5));
    }
}

#[spirv(compute(threads(8, 8)))]
pub fn temporal_filter_cs(
    #[spirv(descriptor_set = 0, binding = 0)] input_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] history_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] reprojection_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] final_output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(descriptor_set = 0, binding = 4)] history_output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(uniform, descriptor_set = 0, binding = 5)] output_tex_size: &Vec4,
    #[spirv(descriptor_set = 0, binding = 32)] sampler_lnc: &Sampler,
    #[spirv(global_invocation_id)] px: UVec3,
) {
    let uv = get_uv_u(px.xy(), *output_tex_size);
    let center: Vec4 = input_tex.fetch(px.xy());

    let reproj: Vec4 = reprojection_tex.fetch(px.xy());
    let history: Vec4 = history_tex.sample_by_lod(*sampler_lnc, uv + reproj.xy(), 0.0);

    let mut vsum = Vec4::ZERO;
    let mut vsum2 = Vec4::ZERO;
    let mut wsum = 0.0;

    let k = 2i32;
    for y in -k..=k {
        for x in -k..=k {
            let neigh: Vec4 = input_tex.fetch(px.xy().as_ivec2() + ivec2(x, y) * 2);
            let w = (-3.0 * (x * x + y * y) as f32 / ((k + 1) * (k + 1)) as f32).exp();
            vsum += neigh * w;
            vsum2 += neigh * neigh * w;
            wsum += w;
        }
    }

    let ex = vsum / wsum;
    let ex2 = vsum2 / wsum;
    let dev = (Vec4::ZERO.max(ex2 - ex * ex)).sqrt();

    let box_size = 0.5;

    let n_deviations = 5.0;
    let nmin = center.lerp(ex, box_size * box_size) - dev * box_size * n_deviations;
    let nmax = center.lerp(ex, box_size * box_size) + dev * box_size * n_deviations;

    let clamped_history = history.clamp(nmin, nmax);
    let res = clamped_history.lerp(center, 1.0.lerp(1.0 / 8.0, reproj.z));

    unsafe {
        history_output_tex.write(px.truncate(), res);
        final_output_tex.write(px.truncate(), res);
    }
}

fn fetch_lighting(
    uv: Vec2,
    input_tex_size: Vec2,
    reprojection_tex: &Image!(2D, type=f32, sampled=true),
    prev_radiance_tex: &Image!(2D, type=f32, sampled=true),
) -> Vec3 {
    let px = (input_tex_size * uv).as_uvec2();
    let reproj: Vec4 = reprojection_tex.fetch(px);
    let prev_rad: Vec4 = prev_radiance_tex.fetch((input_tex_size * (uv + reproj.xy())).as_uvec2());

    Vec3::ZERO.lerp(prev_rad.xyz(), reproj.z)
}

fn fetch_normal_vs(
    uv: Vec2,
    output_tex_size: Vec2,
    view_normal_tex: &Image!(2D, type=f32, sampled=true),
) -> Vec3 {
    let px = (output_tex_size * uv).as_uvec2();
    let normal_vs: Vec4 = view_normal_tex.fetch(px);
    (normal_vs.xyz() * 2.0) - Vec3::splat(1.0)
    //normal_vs.xyz()
}

fn integrate_half_arc(h1: f32, n: f32) -> f32 {
    let a = -(2.0 * h1 - n).cos() + n.cos() + 2.0 * h1 * n.sin();
    0.25 * a
}

fn integrate_arc(h1: f32, h2: f32, n: f32) -> f32 {
    let a = -(2.0 * h1 - n).cos() + n.cos() + 2.0 * h1 * n.sin();
    let b = -(2.0 * h2 - n).cos() + n.cos() + 2.0 * h2 * n.sin();
    0.25 * (a + b)
}

fn update_horizion_angle(prev: f32, cur: f32, blend: f32) -> f32 {
    if cur > prev {
        FloatExt::lerp(prev, cur, blend)
    } else {
        prev
    }
}

fn intersect_dir_plane_onesided(dir: Vec3, normal: Vec3, pt: Vec3) -> f32 {
    let d = -pt.dot(normal);
    d / 1e-5f32.max(-dir.dot(normal))
}

fn process_ssgi_sample(
    frame_constants: &FrameConstants,
    reprojection_tex: &Image!(2D, type=f32, sampled=true),
    prev_radiance_tex: &Image!(2D, type=f32, sampled=true),
    view_normal_tex: &Image!(2D, type=f32, sampled=true),
    input_tex_size: Vec2,
    output_tex_size: Vec2,
    i: u32,
    intsgn: f32,
    n_angle: f32,
    prev_sample_vs: &mut Vec3,
    sample_cs: Vec4,
    center_vs: Vec3,
    normal_vs: Vec3,
    v_vs: Vec3,
    kernel_radius_vs: f32,
    theta_cos_max: f32,
    color_accum: &mut Vec4,
) -> f32 {
    let mut theta_cos_max = theta_cos_max;
    let mut n_angle = n_angle;

    if sample_cs.z > 0.0 {
        let sample_vs4 = frame_constants.view_constants.sample_to_view * sample_cs;
        let sample_vs = sample_vs4.xyz() / sample_vs4.w;
        let sample_vs_offset = sample_vs - center_vs;
        let sample_vs_offset_len = sample_vs_offset.length();

        let sample_theta_cos = sample_vs_offset.dot(v_vs) / sample_vs_offset_len;
        let sample_distance_normalized = sample_vs_offset_len / kernel_radius_vs;

        if sample_distance_normalized < 1.0 {
            let sample_influence = 1.0 - sample_distance_normalized * sample_distance_normalized;

            let sample_visible = sample_theta_cos >= theta_cos_max;
            let theta_cos_prev = theta_cos_max;
            theta_cos_max =
                update_horizion_angle(theta_cos_max, sample_theta_cos, sample_influence);

            if sample_visible {
                let mut lighting = fetch_lighting(
                    cs_to_uv(sample_cs.xy()),
                    input_tex_size,
                    reprojection_tex,
                    prev_radiance_tex,
                );

                let sample_normal_vs =
                    fetch_normal_vs(cs_to_uv(sample_cs.xy()), output_tex_size, view_normal_tex);
                let mut theta_cos_prev_trunc = theta_cos_prev;

                // Account for the sampled surface's normal, and how it's facing the center pixel
                if i > 0 {
                    let p1 = *prev_sample_vs
                        * intersect_dir_plane_onesided(
                            *prev_sample_vs,
                            sample_normal_vs,
                            sample_vs,
                        )
                        .min(intersect_dir_plane_onesided(
                            *prev_sample_vs,
                            normal_vs,
                            center_vs,
                        ));

                    theta_cos_prev_trunc = (p1 - center_vs)
                        .normalize()
                        .dot(v_vs)
                        .clamp(theta_cos_prev_trunc, theta_cos_max);
                }

                // Scale the lighting contribution by the cosine factor
                {
                    n_angle *= -intsgn;

                    let h1 = fast_acos(theta_cos_prev_trunc);
                    let h2 = fast_acos(theta_cos_max);

                    let h1p = n_angle + (h1 - n_angle).max(-core::f32::consts::FRAC_PI_2);
                    let h2p = n_angle + (h2 - n_angle).min(core::f32::consts::FRAC_PI_2);

                    let inv_ao =
                        integrate_half_arc(h1p, n_angle) - integrate_half_arc(h2p, n_angle);

                    lighting *= inv_ao;
                    lighting *= 0.0.step((-sample_vs_offset.normalize()).dot(sample_normal_vs));
                }

                *color_accum += lighting.extend(1.0);
            }
        }

        *prev_sample_vs = sample_vs;
    } else {
        // Sky; assume no occlusion
        theta_cos_max = update_horizion_angle(theta_cos_max, -1.0, 1.0);
    }

    theta_cos_max
}

const TEMPORAL_ROTATIONS: [f32; 6] = [60.0, 300.0, 180.0, 240.0, 120.0, 0.0];
const TEMPORAL_OFFSETS: [f32; 4] = [0.0, 0.5, 0.25, 0.75];

#[spirv(compute(threads(8, 8)))]
pub fn ssgi_cs(
    #[spirv(descriptor_set = 0, binding = 0)] gbuffer_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] half_depth_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] view_normal_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] prev_radiance_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] reprojection_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] output_tex: &Image!(2D, type=f32, sampled=false),
    #[spirv(uniform, descriptor_set = 0, binding = 6)] constants: &SsgiConstants,
    #[spirv(uniform, descriptor_set = 2, binding = 0)] frame_constants: &FrameConstants,
    #[spirv(global_invocation_id)] px: UVec3,
) {
    /* Settings */

    let uv = get_uv_u(px.xy(), constants.output_tex_size);

    let depth: Vec4 = half_depth_tex.fetch(px.xy());
    let depth = depth.x;
    if depth == 0.0 {
        unsafe {
            output_tex.write(px.xy(), vec4(0.0, 0.0, 0.0, 1.0));
        }
        return;
    }

    let gbuffer_packed: Vec4 = gbuffer_tex.fetch(px.xy() * 2);
    let gbuffer = GbufferDataPacked::from(gbuffer_packed.to_bits()).unpack();
    let normal_vs = (frame_constants.view_constants.world_to_view * gbuffer.normal.extend(0.0))
        .xyz()
        .normalize();

    let kernel_radius_cs = ssgi_kernel_radius(constants);

    let view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth, frame_constants);
    let v_vs = -view_ray_context.ray_dir_vs().normalize();

    let ray_hit_cs = view_ray_context.ray_hit_cs;
    let ray_hit_vs = view_ray_context.ray_hit_vs();

    let mut spatial_direction_noise = 1.0 / 16.0 * ((((px.x + px.y) & 3) << 2) + (px.x & 3)) as f32;
    let temporal_direction_noise =
        TEMPORAL_ROTATIONS[(frame_constants.frame_index % 6) as usize] / 360.0;
    let spatial_offset_noise = (1.0 / 4.0) * ((px.y - px.x) & 3) as f32;
    let temporal_offset_noise = TEMPORAL_OFFSETS[(frame_constants.frame_index / 6 % 4) as usize];

    if constants.use_random_jitter {
        let seed0 = hash3(uvec3(frame_constants.frame_index, px.x, px.y));
        spatial_direction_noise += uint_to_u01_float(seed0) * 0.1;
    }

    let ss_angle =
        (spatial_direction_noise + temporal_direction_noise).fract() * core::f32::consts::PI;
    let rand_offset = (spatial_offset_noise + temporal_offset_noise).fract();

    let mut cs_slice_dir = vec2(
        ss_angle.cos() * constants.input_tex_size.y / constants.input_tex_size.x,
        ss_angle.sin(),
    );

    let kernel_radius_shrinkage = {
        // Convert AO radius into world scale
        let cs_kernel_radius_scaled = if constants.use_kernel_distance_scaling {
            kernel_radius_cs
                * frame_constants
                    .view_constants
                    .view_to_clip
                    .to_cols_array_2d()[1][1]
                / -ray_hit_vs.z
        } else {
            kernel_radius_cs
        };

        cs_slice_dir *= cs_kernel_radius_scaled;

        // Calculate AO radius shrinkage (if camera is too close to a surface)
        let max_kernel_radius_cs = constants.max_kernel_radius_cs;
        1.0f32.min(max_kernel_radius_cs / cs_kernel_radius_scaled)
    };

    // Shrink the AO radius
    cs_slice_dir *= kernel_radius_shrinkage;
    let kernel_radius_vs = kernel_radius_cs * kernel_radius_shrinkage * -ray_hit_vs.z;

    let center_vs = ray_hit_vs.xyz();

    cs_slice_dir *= 1.0 / constants.ssgi_half_sample_count as f32;
    let vs_slice_dir =
        (frame_constants.view_constants.sample_to_view * cs_slice_dir.extend(0.0).extend(0.0)).xy();
    let slice_normal_vs = v_vs.cross(vs_slice_dir.extend(0.0)).normalize();

    let mut proj_normal_vs = normal_vs - slice_normal_vs * slice_normal_vs.dot(normal_vs);
    let slice_contrib_weight = proj_normal_vs.length();
    proj_normal_vs /= slice_contrib_weight;

    let n_angle = fast_acos(proj_normal_vs.dot(v_vs).clamp(-1.0, 1.0))
        * sign(vs_slice_dir.dot(proj_normal_vs.xy() - v_vs.xy()));

    let mut theta_cos_max1 = (n_angle - core::f32::consts::FRAC_PI_2).cos();
    let mut theta_cos_max2 = (n_angle + core::f32::consts::FRAC_PI_2).cos();

    let mut color_accum = Vec4::ZERO;

    let mut prev_sample0_vs = v_vs;
    let mut prev_sample1_vs = v_vs;

    let mut prev_sample_coord0 = px.xy();
    let mut prev_sample_coord1 = px.xy();

    for i in 0..constants.ssgi_half_sample_count {
        {
            let t = i as f32 + rand_offset;

            let mut sample_cs = (ray_hit_cs.xy() - cs_slice_dir * t).extend(0.0).extend(1.0);
            let sample_px = (constants.output_tex_size.xy() * cs_to_uv(sample_cs.xy())).as_uvec2();

            if sample_px != prev_sample_coord0 {
                prev_sample_coord0 = sample_px;
                let half_depth: Vec4 = half_depth_tex.fetch(sample_px);
                sample_cs.z = half_depth.x;
                theta_cos_max1 = process_ssgi_sample(
                    frame_constants,
                    reprojection_tex,
                    prev_radiance_tex,
                    view_normal_tex,
                    constants.input_tex_size.xy(),
                    constants.output_tex_size.xy(),
                    i,
                    1.0,
                    n_angle,
                    &mut prev_sample0_vs,
                    sample_cs,
                    center_vs,
                    normal_vs,
                    v_vs,
                    kernel_radius_vs,
                    theta_cos_max1,
                    &mut color_accum,
                );
            }
        }
        {
            let t = i as f32 + (1.0 - rand_offset);

            let mut sample_cs = (ray_hit_cs.xy() + cs_slice_dir * t).extend(0.0).extend(1.0);
            let sample_px = (constants.output_tex_size.xy() * cs_to_uv(sample_cs.xy())).as_uvec2();

            if sample_px != prev_sample_coord1 {
                prev_sample_coord1 = sample_px;
                let half_depth: Vec4 = half_depth_tex.fetch(sample_px);
                sample_cs.z = half_depth.x;
                theta_cos_max2 = process_ssgi_sample(
                    frame_constants,
                    reprojection_tex,
                    prev_radiance_tex,
                    view_normal_tex,
                    constants.input_tex_size.xy(),
                    constants.output_tex_size.xy(),
                    i,
                    -1.0,
                    n_angle,
                    &mut prev_sample1_vs,
                    sample_cs,
                    center_vs,
                    normal_vs,
                    v_vs,
                    kernel_radius_vs,
                    theta_cos_max2,
                    &mut color_accum,
                );
            }
        }
    }

    let h1 = -fast_acos(theta_cos_max1);
    let h2 = fast_acos(theta_cos_max2);

    let h1p = n_angle + (h1 - n_angle).max(-core::f32::consts::FRAC_PI_2);
    let h2p = n_angle + (h2 - n_angle).min(core::f32::consts::FRAC_PI_2);

    let inv_ao = integrate_arc(h1p, h2p, n_angle);

    let mut col = if constants.use_ao_only {
        Vec4::splat(0.0f32.max(inv_ao))
    } else {
        color_accum.xyz().extend(0.0f32.max(inv_ao))
    };

    col *= slice_contrib_weight;

    /*float bent_normal_angle = h1p + h2p - n_angle * 2;
    float3 bent_normal_dir = sin(bent_normal_angle) * cross(slice_normal_vs, normal_vs) + cos(bent_normal_angle) * normal_vs;
    bent_normal_dir = bent_normal_dir;*/

    unsafe {
        output_tex.write(px.truncate(), col);
    }
}
