use macaw::{Vec2, Vec4, Mat4, Mat2, UVec4};

#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
pub struct MeshDescriptor {
    pub vertex_core_offset: u32,      // position, normal packed in one
    pub vertex_prev_core_offset: u32, // previous position, noraml packed in one. Usually 0, if not skinning.
    pub vertex_uv_offset: u32,
    pub vertex_mat_offset: u32,
    pub vertex_aux_offset: u32,
    pub vertex_tangent_offset: u32,
    pub vertex_bone_indices_weights_offset: u32,
    pub index_offset: u32,
    pub flags: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct InstanceTransform {
    pub transform: Mat4,
    pub prev_transform: Mat4,
}

#[derive(Clone, Copy)]
#[repr(C)]
#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
pub struct TextureMaps(UVec4);

impl TextureMaps {
    #[inline(always)]
    pub fn normal(&self) -> usize {
        self.0.x as usize
    }

    #[inline(always)]
    pub fn metallic_roughness(&self) -> usize {
        self.0.y as usize
    }

    #[inline(always)]
    pub fn albedo(&self) -> usize {
        self.0.z as usize
    }

    #[inline(always)]
    pub fn emissive(&self) -> usize {
        self.0.w as usize
    }

}

#[derive(Clone, Copy, Default)]
pub struct TextureMapsBuilder(UVec4);

impl TextureMapsBuilder {
    pub fn new() -> Self { Default::default() }

    pub fn with_normal(mut self, normal: u32) -> Self {
        self.0.x = normal;
        self
    }

    pub fn with_metallic_roughness(mut self, metallic_roughness: u32) -> Self {
        self.0.y = metallic_roughness;
        self
    }

    pub fn with_albedo(mut self, albedo: u32) -> Self {
        self.0.z = albedo;
        self
    }

    pub fn with_emissive(mut self, emissive: u32) -> Self {
        self.0.w = emissive;
        self
    }

    pub fn build(self) -> TextureMaps {
        TextureMaps(self.0)
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
pub struct MaterialDescriptor {
    pub base_color_mult: Vec4,
    pub maps: TextureMaps,
    pub roughness_mult: f32,
    pub metalness_factor: f32,
    pub emissive: Vec4,
    pub flags: u32,
    pub map_transforms: [[f32; 6]; 4],
}

impl MaterialDescriptor {
    pub fn load(data: &[u32], byte_offset: u32) -> Self {
        let offset = (byte_offset >> 2) as usize;
        let base_color_mult = load_vec4(data, offset);
        let maps = TextureMaps(UVec4::new(
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ));
        let roughness_mult = f32::from_bits(data[offset + 8]);
        let metalness_factor = f32::from_bits(data[offset + 9]);
        let emissive = load_vec4(data, offset + 10);
        let flags = data[offset + 15];
        let map_transforms = load_map_transforms(data, offset + 16);

        Self {
            base_color_mult,
            maps,
            roughness_mult,
            metalness_factor,
            emissive,
            flags,
            map_transforms,
        }
    }
    pub fn transform_uv(&self, uv: Vec2, map_idx: usize) -> Vec2 {
        let mat = &self.map_transforms[map_idx];
        let rot_scl: Mat2 = Mat2::from_cols(Vec2::new(mat[0], mat[2]), Vec2::new(mat[1], mat[3]));
        let offset: Vec2 = Vec2::new(mat[4], mat[5]);
        rot_scl * uv + offset
    }

}

fn load_vec4(data: &[u32], offset: usize) -> Vec4 {
    Vec4::new(
        f32::from_bits(data[offset]),
        f32::from_bits(data[offset + 1]),
        f32::from_bits(data[offset + 2]),
        f32::from_bits(data[offset + 3]),
    )
}

fn load_f32_6(data: &[u32], offset: usize) -> [f32; 6] {
    [
        f32::from_bits(data[offset]),
        f32::from_bits(data[offset + 1]),
        f32::from_bits(data[offset + 2]),
        f32::from_bits(data[offset + 3]),
        f32::from_bits(data[offset + 4]),
        f32::from_bits(data[offset + 5]),
    ]
}

fn load_map_transforms(data: &[u32], offset: usize) -> [[f32; 6]; 4] {
    [
        load_f32_6(data, offset),
        load_f32_6(data, offset + 6),
        load_f32_6(data, offset + 12),
        load_f32_6(data, offset + 18),
    ]
}
