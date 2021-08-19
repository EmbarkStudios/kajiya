use crate::math::*;

pub trait IntoCameraBodyMatrices {
    fn into_camera_body_matrices(self) -> CameraBodyMatrices;
}

impl IntoCameraBodyMatrices for CameraBodyMatrices {
    fn into_camera_body_matrices(self) -> CameraBodyMatrices {
        self
    }
}

impl IntoCameraBodyMatrices for (Vec3, Quat) {
    fn into_camera_body_matrices(self) -> CameraBodyMatrices {
        CameraBodyMatrices::from_position_rotation(self.0, self.1)
    }
}

pub trait LookThroughCamera {
    fn through(self, lens: &CameraLens) -> CameraMatrices;
}

impl<T> LookThroughCamera for T
where
    T: IntoCameraBodyMatrices,
{
    fn through(self, lens: &CameraLens) -> CameraMatrices {
        let body = self.into_camera_body_matrices();
        let lens = lens.calc_matrices();
        CameraMatrices {
            view_to_clip: lens.view_to_clip,
            clip_to_view: lens.clip_to_view,
            world_to_view: body.world_to_view,
            view_to_world: body.view_to_world,
        }
    }
}

#[derive(Clone, Copy)]
pub struct CameraLens {
    pub near_plane_distance: f32,
    pub aspect_ratio: f32,
    pub vertical_fov: f32,
}

impl Default for CameraLens {
    fn default() -> Self {
        Self {
            near_plane_distance: 0.01, // 1mm
            aspect_ratio: 1.0,
            vertical_fov: 52.0,
        }
    }
}

pub struct CameraLensMatrices {
    pub view_to_clip: Mat4,
    pub clip_to_view: Mat4,
}

#[derive(PartialEq, Clone, Copy)]
pub struct CameraBodyMatrices {
    pub world_to_view: Mat4,
    pub view_to_world: Mat4,
}

#[derive(PartialEq, Clone, Copy)]
pub struct CameraMatrices {
    pub view_to_clip: Mat4,
    pub clip_to_view: Mat4,
    pub world_to_view: Mat4,
    pub view_to_world: Mat4,
}

impl CameraBodyMatrices {
    pub fn from_position_rotation(position: Vec3, rotation: Quat) -> Self {
        let view_to_world = {
            let translation = Mat4::from_translation(position);
            translation * Mat4::from_quat(rotation)
        };

        let world_to_view = {
            let inv_translation = Mat4::from_translation(-position);
            Mat4::from_quat(rotation.conjugate()) * inv_translation
        };

        Self {
            world_to_view,
            view_to_world,
        }
    }
}

impl CameraMatrices {
    pub fn eye_position(&self) -> Vec3 {
        (self.view_to_world * Vec4::new(0.0, 0.0, 0.0, 1.0)).truncate()
    }

    pub fn eye_direction(&self) -> Vec3 {
        (self.view_to_world * Vec4::new(0.0, 0.0, -1.0, 0.0))
            .truncate()
            .normalize()
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.view_to_clip.y_axis.y / self.view_to_clip.x_axis.x
    }
}

impl CameraLens {
    fn calc_matrices(&self) -> CameraLensMatrices {
        let fov = self.vertical_fov.to_radians();
        let znear = self.near_plane_distance;

        let h = (0.5 * fov).cos() / (0.5 * fov).sin();
        let w = h / self.aspect_ratio;

        /*let mut m = Mat4::ZERO;
        m.m11 = w;
        m.m22 = h;
        m.m34 = znear;
        m.m43 = -1.0;
        m*/
        let view_to_clip = Mat4::from_cols(
            Vec4::new(w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, znear, 0.0),
        );

        /*let mut m = Mat4::ZERO;
        m.m11 = 1.0 / w;
        m.m22 = 1.0 / h;
        m.m34 = -1.0;
        m.m43 = 1.0 / znear;
        m*/
        let clip_to_view = Mat4::from_cols(
            Vec4::new(1.0 / w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0 / h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0 / znear),
            Vec4::new(0.0, 0.0, -1.0, 0.0),
        );

        CameraLensMatrices {
            view_to_clip,
            clip_to_view,
        }
    }
}
