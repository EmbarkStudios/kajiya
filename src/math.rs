pub use glam::{Mat2, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

#[allow(dead_code)]
pub fn build_orthonormal_basis(n: Vec3) -> Mat3 {
    let b1;
    let b2;

    if n.z() < 0.0 {
        let a = 1.0 / (1.0 - n.z());
        let b = n.x() * n.y() * a;
        b1 = Vec3::new(1.0 - n.x() * n.x() * a, -b, n.x());
        b2 = Vec3::new(b, n.y() * n.y() * a - 1.0, -n.y());
    } else {
        let a = 1.0 / (1.0 + n.z());
        let b = -n.x() * n.y() * a;
        b1 = Vec3::new(1.0 - n.x() * n.x() * a, b, -n.x());
        b2 = Vec3::new(b, 1.0 - n.y() * n.y() * a, -n.y());
    }

    Mat3::from_cols(b1, b2, n)
}
