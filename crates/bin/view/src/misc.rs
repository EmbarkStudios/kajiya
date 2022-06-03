#![allow(dead_code)]

pub fn smoothstep(edge0: f32, edge1: f32, mut x: f32) -> f32 {
    // Scale, bias and saturate x to 0..1 range
    x = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    // Evaluate polynomial
    x * x * (3.0 - 2.0 * x)
}
