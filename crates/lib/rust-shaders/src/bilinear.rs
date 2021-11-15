use macaw::*;

pub struct Bilinear {
    pub origin: Vec2,
    pub weights: Vec2,
}

impl Bilinear {
    pub fn new(uv: Vec2, tex_size: Vec2) -> Self {
        Self {
            origin: (uv * tex_size - 0.5).trunc(),
            weights: (uv * tex_size - 0.5).fract(),
        }
    }

    pub fn px0(&self) -> IVec2 {
        self.origin.as_ivec2()
    }

    pub fn px1(&self) -> IVec2 {
        self.origin.as_ivec2() + ivec2(1, 0)
    }

    pub fn px2(&self) -> IVec2 {
        self.origin.as_ivec2() + ivec2(0, 1)
    }

    pub fn px3(&self) -> IVec2 {
        self.origin.as_ivec2() + ivec2(1, 1)
    }

    pub fn custom_weights(&self, custom_weights: Vec4) -> Vec4 {
        let weights = vec4(
            (1.0 - self.weights.x) * (1.0 - self.weights.y),
            self.weights.x * (1.0 - self.weights.y),
            (1.0 - self.weights.x) * self.weights.y,
            self.weights.x * self.weights.y,
        );

        weights * custom_weights
    }
}

pub fn apply_bilinear_custom_weights(
    s00: Vec4,
    s10: Vec4,
    s01: Vec4,
    s11: Vec4,
    w: Vec4,
    normalize: bool,
) -> Vec4 {
    let r = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    if normalize {
        r * w.dot(Vec4::ONE).recip()
    } else {
        r
    }
}
