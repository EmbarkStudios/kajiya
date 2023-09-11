use glam::{UVec4, Vec2, Vec3};
use rust_shaders_shared::util;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GBufferData {
    pub albedo: Vec3,
    pub emissive: Vec3,
    pub normal: Vec3,
    pub roughness: f32,
    pub metalness: f32,
}

impl GBufferData {
    pub fn pack(self) -> UVec4 {
        let mut res: UVec4 = Default::default();

        res.x = util::pack_color_888(self.albedo);
        res.y = util::pack_normal_11_10_11(self.normal) as u32;

        let roughness_metalness = Vec2::new(
            util::roughness_to_perceptual_roughness(self.roughness),
            self.metalness,
        );

        res.z = spirv_std::float::vec2_to_f16x2(roughness_metalness);
        res.w = util::float3_to_rgb9e5(self.emissive);

        res
    }
}
