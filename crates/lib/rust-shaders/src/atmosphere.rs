// not used

// Derived from atmosphere_felix.hlsl.

use core::f32::consts::PI;
use macaw::{const_vec3, Vec2, Vec3};

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

const PLANET_RADIUS: f32 = 6371000.0;
const PLANET_CENTER: Vec3 = const_vec3!([0.0, -PLANET_RADIUS, 0.0]);
const ATMOSPHERE_HEIGHT: f32 = 100000.0;
const RAYLEIGH_HEIGHT: f32 = ATMOSPHERE_HEIGHT * 0.08;
const MIE_HEIGHT: f32 = ATMOSPHERE_HEIGHT * 0.012;

const C_RAYLEIGH: Vec3 = const_vec3!([5.802 * 1e-6, 13.558 * 1e-6, 33.100 * 1e-6]);
const C_MIE: Vec3 = const_vec3!([3.996 * 1e-6, 3.996 * 1e-6, 3.996 * 1e-6]);
const C_OZONE: Vec3 = const_vec3!([0.650 * 1e-6, 1.881 * 1e-6, 0.085 * 1e-6]);

const ATMOSPHERE_DENSITY: f32 = 1.0;
const EXPOSURE: f32 = 20.0;

/// Optical depth is a unitless measurement of the amount of absorption of a participating medium (such as the atmosphere).
/// This function calculates just that for our three atmospheric elements:
/// R: Rayleigh
/// G: Mie
/// B: Ozone
/// If you find the term "optical depth" confusing, you can think of it as "how much density was found along the ray in total".
pub fn integrate_optical_depth(ray_o: Vec3, ray_d: Vec3) -> Vec3 {
    let intersection = atmosphere_intersection(ray_o, ray_d);
    let ray_length = intersection.y;

    let sample_count = 8;
    let step_size = ray_length / sample_count as f32;

    let mut optical_depth = Vec3::ZERO;

    let mut i = 0;
    // Using a while loop here as a workaround for a spirv-cross bug.
    // See https://github.com/EmbarkStudios/rust-gpu/issues/739
    while i < sample_count {
        let local_pos = ray_o + ray_d * (i as f32 + 0.5) * step_size;
        let local_height = atmosphere_height(local_pos);
        let local_density = atmosphere_density(local_height);

        optical_depth += local_density * step_size;

        i += 1;
    }

    optical_depth
}

pub fn atmosphere_height(position_ws: Vec3) -> f32 {
    (position_ws - PLANET_CENTER).length() - PLANET_RADIUS
}

fn density_rayleigh(h: f32) -> f32 {
    (-(0.0f32.max(h / RAYLEIGH_HEIGHT))).exp()
}

fn density_mie(h: f32) -> f32 {
    (-(0.0f32.max(h / MIE_HEIGHT))).exp()
}

fn density_ozone(h: f32) -> f32 {
    // The ozone layer is represented as a tent function with a width of 30km, centered around an altitude of 25km.
    0.0f32.max(1.0 - (h - 25000.0).abs() / 15000.0)
}

pub fn atmosphere_density(h: f32) -> Vec3 {
    Vec3::new(density_rayleigh(h), density_mie(h), density_ozone(h))
}

pub fn sphere_intersection(mut ray_o: Vec3, ray_d: Vec3, sphere_o: Vec3, sphere_r: f32) -> Vec2 {
    ray_o -= sphere_o;
    let a = ray_d.dot(ray_d);
    let b = 2.0 * ray_o.dot(ray_d);
    let c = ray_o.dot(ray_o) - sphere_r * sphere_r;
    let d = b * b - 4.0 * a * c;
    if d < 0.0 {
        Vec2::splat(-1.0)
    } else {
        let d = d.sqrt();
        Vec2::new(-b - d, -b + d) / (2.0 * a)
    }
}

pub fn atmosphere_intersection(ray_o: Vec3, ray_d: Vec3) -> Vec2 {
    sphere_intersection(
        ray_o,
        ray_d,
        PLANET_CENTER,
        PLANET_RADIUS + ATMOSPHERE_HEIGHT,
    )
}

// -------------------------------------
// Phase functions
fn phase_rayleigh(costh: f32) -> f32 {
    3.0 * (1.0 + costh * costh) / (16.0 * PI)
}

fn phase_mie(costh: f32, mut g: f32) -> f32 {
    // g = 0.85)
    g = g.min(0.9381);
    let k = 1.55 * g - 0.55 * g * g * g;
    let kcosth = k * costh;
    (1.0 - k * k) / ((4.0 * PI) * (1.0 - kcosth) * (1.0 - kcosth))
}

/// Calculate a luminance transmittance value from optical depth.
pub fn absorb(optical_depth: Vec3) -> Vec3 {
    // Note that Mie results in slightly more light absorption than scattering, about 10%
    (-(optical_depth.x * C_RAYLEIGH + optical_depth.y * C_MIE * 1.1 + optical_depth.z * C_OZONE)
        * ATMOSPHERE_DENSITY)
        .exp()
}

// Integrate scattering over a ray for a single directional light source.
// Also return the transmittance for the same ray as we are already calculating the optical depth anyway.
pub fn integrate_scattering(
    mut ray_start: Vec3,
    ray_dir: Vec3,
    mut ray_length: f32,
    light_dir: Vec3,
    light_color: Vec3,
    transmittance: &mut Vec3,
) -> Vec3 {
    // We can reduce the number of atmospheric samples required to converge by spacing them exponentially closer to the camera.
    // This breaks space view however, so let's compensate for that with an exponent that "fades" to 1 as we leave the atmosphere.
    // let ray_height = atmosphere_height(ray_start);
    //float  sample_distribution_exponent = 1 + saturate(1 - ray_height / ATMOSPHERE_HEIGHT) * 8; // Slightly arbitrary max exponent of 9
    //float  sample_distribution_exponent = 1 + 8 * abs(ray_dir.y);
    let sample_distribution_exponent: f32 = 5.0;

    let intersection: Vec2 = atmosphere_intersection(ray_start, ray_dir);

    ray_length = ray_length.min(intersection.y);
    if intersection.x > 0.0 {
        // Advance ray to the atmosphere entry point
        ray_start += ray_dir * intersection.x;
        ray_length -= intersection.x;
    }

    let costh = ray_dir.dot(light_dir);
    let phase_r = phase_rayleigh(costh);
    let phase_m = phase_mie(costh, 0.85);

    let sample_count: usize = 16;

    let mut optical_depth = Vec3::ZERO;
    let mut rayleigh = Vec3::ZERO;
    let mut mie = Vec3::ZERO;

    let mut prev_ray_time = 0.0f32;

    for i in 1..=sample_count {
        let ray_time: f32 =
            (i as f32 / sample_count as f32).powf(sample_distribution_exponent) * ray_length;
        // Because we are distributing the samples exponentially, we have to calculate the step size per sample.
        let step_size = ray_time - prev_ray_time;

        //float3 local_position = ray_start + ray_dir * ray_time;
        let local_position: Vec3 =
            ray_start + ray_dir * macaw::FloatExt::lerp(prev_ray_time, ray_time, 0.5);
        let local_height: f32 = atmosphere_height(local_position);
        let local_density: Vec3 = atmosphere_density(local_height);

        optical_depth += local_density * step_size;

        // The atmospheric transmittance from ray_start to local_position
        let view_transmittance: Vec3 = absorb(optical_depth);

        let optical_depthlight: Vec3 = integrate_optical_depth(local_position, light_dir);

        // The atmospheric transmittance of light reaching local_position
        let light_transmittance: Vec3 = absorb(optical_depthlight);

        rayleigh +=
            view_transmittance * light_transmittance * phase_r * local_density.x * step_size;
        mie += view_transmittance * light_transmittance * phase_m * local_density.y * step_size;

        prev_ray_time = ray_time;
    }

    *transmittance = absorb(optical_depth);

    (rayleigh * C_RAYLEIGH + mie * C_MIE) * light_color * EXPOSURE
}
