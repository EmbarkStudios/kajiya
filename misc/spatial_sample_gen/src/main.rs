use std::fs::File;
use std::io::prelude::*;

use image::{io::Reader as ImageReader, Rgb};

fn main() {
    const TARGET_SAMPLE_COUNT: usize = 16;

    let mut out_samples = File::create("out_samples.rs").unwrap();
    writeln!(
        out_samples,
        "pub const SPATIAL_RESOLVE_SAMPLES: [(i32, i32, i32, i32); {} * 4 * 8] = [",
        TARGET_SAMPLE_COUNT
    )
    .unwrap();

    let kernel_count = 8;
    for kernel_i in 0..kernel_count {
        let mut img = ImageReader::open(format!("LDR_LLL1_{}.png", kernel_i))
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();

        let cutoff = {
            let t = kernel_i as f32 / (kernel_count - 1) as f32;
            ((1.0 + (0.125 - 1.0) * t.sqrt()) * 64.0) as i32
        };

        let is_within_group = |px: u8, group: i32| -> bool {
            let px = px as i32;
            px >= cutoff * group && px < cutoff * (group + 1)
        };

        let mut groups: [Vec<(u32, u32)>; 4] = Default::default();
        let center = 15i32;

        for (x, y, px) in img.enumerate_pixels() {
            for group in 0..4 {
                if (x as i32 == center && y as i32 == center)
                    || is_within_group(px[0], group as i32)
                {
                    groups[group].push((x, y));
                }
            }
        }

        for group in &mut groups {
            group.sort_by_key(|(x, y)| {
                ((*x as i32) - center).pow(2) + ((*y as i32) - center).pow(2)
            });
            group.truncate(TARGET_SAMPLE_COUNT);
            assert!(group.len() == TARGET_SAMPLE_COUNT);

            for (x, y) in group {
                let xo = *x as i32 - center;
                let yo = *y as i32 - center;
                writeln!(out_samples, "    ({}i32, {}i32, 0, 0),", xo, yo).unwrap();
            }
        }

        for (x, y, px) in img.enumerate_pixels_mut() {
            if x as i32 == center && y as i32 == center {
                *px = Rgb([255, 255, 255]);
            } else if groups[0].contains(&(x, y)) {
                *px = Rgb([255, 0, 0]);
            } else if groups[1].contains(&(x, y)) {
                *px = Rgb([20, 200, 100]);
            } else if groups[2].contains(&(x, y)) {
                *px = Rgb([0, 0, 255]);
            } else if groups[3].contains(&(x, y)) {
                *px = Rgb([255, 255, 0]);
            } else {
                *px = Rgb([0, 0, 0]);
            }
        }

        img.save(format!("out{}.png", kernel_i)).unwrap();
    }

    writeln!(out_samples, "];").unwrap();
}
