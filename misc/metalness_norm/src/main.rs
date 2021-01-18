//! Calculates a look-up table of albedo boost needed to achieve energy
//! preservation when blending between dielectric and metal attributes
//! in a physically-based energy-conserving BRDF.
//!
//! The calculations are done for F0, but remain correct through F90.

use std::fs::File;
use std::io::prelude::*;

use plotters::prelude::*;

#[derive(Clone, Copy, Debug)]
struct Brdf {
    spec_albedo: f64,
    diffuse_albedo: f64,
}

impl Brdf {
    fn calculate(&self) -> f64 {
        self.spec_albedo + (1.0 - self.spec_albedo) * self.diffuse_albedo
    }

    fn with_scaled_albedo(mut self, k: f64) -> Self {
        self.spec_albedo *= k;
        self.diffuse_albedo *= k;
        self
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn find_mult(metalness: f64, diffuse_albedo: f64) -> Option<f64> {
    let spec_albedo = 0.04;

    let dielectric = Brdf {
        spec_albedo: spec_albedo,
        diffuse_albedo: diffuse_albedo,
    };

    let metallic = Brdf {
        spec_albedo: diffuse_albedo,
        diffuse_albedo: 0.0,
    };

    let blend_brdf = Brdf {
        spec_albedo: lerp(spec_albedo, diffuse_albedo, metalness),
        diffuse_albedo: diffuse_albedo * (1.0 - metalness),
    };

    //dbg!(&blend_brdf);

    let reflectance_blend_of_brdfs = lerp(dielectric.calculate(), metallic.calculate(), metalness);

    //dbg!(blend_of_brdfs);
    //dbg!(blended_brdf);

    let mut search_result: Option<f64> = None;
    let mut mult_advance = 0.0625;
    let mut mult = 1.0 / (1.0 + mult_advance);

    for step in 0..1000 {
        let prev_mult = mult;
        mult *= 1.0 + mult_advance;

        let scaled_blend_brdf = blend_brdf.with_scaled_albedo(mult);
        let reflectance_blended_brdf = scaled_blend_brdf.calculate();

        let diff = reflectance_blended_brdf - reflectance_blend_of_brdfs;

        if diff >= 0.0 {
            search_result = Some(mult);
            if diff < 1e-30 {
                //println!("Converged in {} steps; mult: {}, err: {}", step, mult, diff);
                break;
            }

            // Binary search
            mult = prev_mult;
            mult_advance *= 0.5;
        }
    }

    if let Some(mult) = search_result {
        let scaled_blend_brdf = blend_brdf.with_scaled_albedo(mult);
        if scaled_blend_brdf.spec_albedo > 1.0 || scaled_blend_brdf.diffuse_albedo > 1.0 {
            panic!("Albedo exceeded 1.0! {:?}", scaled_blend_brdf);
        }
    }

    search_result
}

use image::ImageBuffer;

const LUT_WIDTH: u32 = 64;
const LUT_HEIGHT: u32 = 64;

fn calculate_lut_texture() {
    // Construct a new by repeated calls to the supplied closure.
    let img = ImageBuffer::from_fn(LUT_WIDTH, LUT_HEIGHT, |x, y| {
        let metalness = x as f64 / (LUT_WIDTH - 1) as f64;
        let diffuse_albedo = y as f64 / LUT_HEIGHT as f64; // don't include 1.0 as it contains a hotspot
                                                           //let diffuse_albedo = diffuse_albedo.sqrt();

        let mult = find_mult(metalness, diffuse_albedo)
            .unwrap_or_else(|| panic!("failed to find mult for metalness {}", metalness));

        assert!(mult < 2.0);

        let value = (mult - 1.0).max(0.0) * 255.0;
        image::Luma([value as u8])
    });

    img.save("metalness_albedo_boost_lut.png").unwrap();
}

fn calculate_lut_csv() {
    let mut file = File::create("metalness_albedo_boost.csv").unwrap();

    for y in 0..LUT_HEIGHT {
        for x in 0..LUT_HEIGHT {
            let metalness = x as f64 / (LUT_WIDTH - 1) as f64;
            let diffuse_albedo = y as f64 / LUT_HEIGHT as f64; // don't include 1.0 as it contains a hotspot
                                                               //let diffuse_albedo = diffuse_albedo.sqrt();

            let mult = find_mult(metalness, diffuse_albedo)
                .unwrap_or_else(|| panic!("failed to find mult for metalness {}", metalness));

            assert!(mult < 2.0);

            let value = (mult - 1.0).max(0.0);

            if x > 0 {
                write!(file, ",");
            }
            write!(file, "{}", value);
        }
        writeln!(file, "");
    }
}

fn main() {
    calculate_lut_texture();
    calculate_lut_csv();
}

const SAMPLE_COUNT: usize = 100;

fn calculate_mults_for_diffuse_albedo(diffuse_albedo: f64, metalness: &[f64]) -> Vec<f64> {
    metalness
        .iter()
        .map(|&metalness| {
            let mult = find_mult(metalness, diffuse_albedo)
                .unwrap_or_else(|| panic!("failed to find mult for metalness {}", metalness));

            assert!(mult < 2.0);

            mult - 1.0
        })
        .collect()
}
fn plot_mult_profile() {
    let diffuse_albedo = 0.9;

    let metalness: Vec<f64> = (0..SAMPLE_COUNT)
        .map(|i| i as f64 / (SAMPLE_COUNT - 1) as f64)
        .collect();
    let mults = calculate_mults_for_diffuse_albedo(diffuse_albedo, &metalness);
    //println!("");

    let root_area = BitMapBackend::new("plot.png", (600, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(LineSeries::new(
        metalness.into_iter().zip(mults.into_iter()),
        &GREEN,
    ))
    .unwrap();
}
