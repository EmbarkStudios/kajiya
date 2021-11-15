use crate::util::*;
use macaw::*;

#[cfg(target_arch = "spirv")]
use spirv_std::float::{f16x2_to_vec2, vec2_to_f16x2};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[repr(C)]
#[derive(Clone)]
pub struct GbufferDataPacked {
    pub v: UVec4,
}

#[derive(Default)]
pub struct GbufferData {
    pub albedo: Vec3,
    pub emissive: Vec3,
    pub normal: Vec3,
    pub roughness: f32,
    pub metalness: f32,
}

pub fn roughness_to_perceptual_roughness(r: f32) -> f32 {
    r.sqrt()
}

pub fn perceptual_roughness_to_roughness(r: f32) -> f32 {
    r * r
}

impl GbufferData {
    #[allow(clippy::unused_self)]
    pub fn pack(&self) -> GbufferDataPacked {
        #[cfg(not(target_arch = "spirv"))]
        return GbufferDataPacked { v: UVec4::ZERO };
        #[cfg(target_arch = "spirv")]
        GbufferDataPacked {
            v: UVec4::new(
                pack_color_888(self.albedo),
                pack_normal_11_10_11(self.normal).to_bits(),
                vec2_to_f16x2(vec2(
                    roughness_to_perceptual_roughness(self.roughness),
                    self.metalness,
                )),
                float3_to_rgb9e5(self.emissive),
            ),
        }
    }
}

impl GbufferDataPacked {
    pub fn unpack(&self) -> GbufferData {
        #[cfg(not(target_arch = "spirv"))]
        let roughness_metalness: Vec2 = Vec2::new(0.0, 0.0);
        #[cfg(target_arch = "spirv")]
        let roughness_metalness: Vec2 = f16x2_to_vec2(self.v.z);

        GbufferData {
            albedo: self.unpack_albedo(),
            emissive: rgb9e5_to_float3(self.v.w),
            normal: self.unpack_normal(),
            roughness: perceptual_roughness_to_roughness(roughness_metalness.x),
            metalness: roughness_metalness.y,
        }
    }

    pub fn unpack_normal(&self) -> Vec3 {
        unpack_normal_11_10_11(f32::from_bits(self.v.y))
    }

    pub fn unpack_albedo(&self) -> Vec3 {
        unpack_color_888(self.v.x)
    }

    pub fn to_vec4(&self) -> Vec4 {
        Vec4::new(
            f32::from_bits(self.v.x),
            f32::from_bits(self.v.y),
            f32::from_bits(self.v.z),
            f32::from_bits(self.v.w),
        )
    }
}

impl From<UVec4> for GbufferDataPacked {
    fn from(data0: UVec4) -> Self {
        Self {
            v: UVec4::new(data0.x, data0.y, data0.z, data0.w),
        }
    }
}
