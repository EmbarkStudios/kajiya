#[allow(non_snake_case)]
pub mod RenderOverrideFlags {
    pub const FORCE_FACE_NORMALS: u32 = 1 << 0;
    pub const NO_NORMAL_MAPS: u32 = 1 << 1;
    pub const FLIP_NORMAL_MAP_YZ: u32 = 1 << 2;
    pub const NO_METAL: u32 = 1 << 3;
}

#[repr(C, align(16))]
#[derive(Copy, Clone, PartialEq)]
pub struct RenderOverrides {
    pub flags: u32,
    pub material_roughness_scale: f32,
}

impl Default for RenderOverrides {
    fn default() -> Self {
        Self {
            flags: 0,
            material_roughness_scale: 1.0,
        }
    }
}

impl RenderOverrides {
    pub fn has_flag(&self, flag: u32) -> bool {
        (self.flags & flag) != 0
    }

    pub fn set_flag(&mut self, flag: u32, value: bool) {
        if value {
            self.flags |= flag;
        } else {
            self.flags &= !flag;
        }
    }
}
