// Cone sweep global illumination prototype

use glam::{Mat3, Vec3};
use rg::GetOrCreateTemporal;
use slingshot::{
    ash::vk,
    backend::{buffer::*, image::*, ray_tracing::RayTracingAcceleration},
    rg::{self, SimpleRenderPass},
    vk_sync, Device,
};

const PRETRACE_DIMS: u32 = 32;

use super::GbufferDepth;

pub struct CsgiRenderer;

pub struct CsgiVolume {
    pub cascade0: rg::Handle<Image>,
    pub alt_cascade0: rg::Handle<Image>,
}

impl CsgiRenderer {
    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
    ) -> CsgiVolume {
        let mut cascade0 = rg
            .get_or_create_temporal(
                "csgi.cascade0",
                ImageDesc::new_3d(
                    vk::Format::R16G16B16A16_SFLOAT,
                    [32 * SLICE_COUNT as u32, 32, 32],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        let mut alt_cascade0 = rg
            .get_or_create_temporal(
                "csgi.alt_cascade0",
                ImageDesc::new_3d(
                    vk::Format::R16G16B16A16_SFLOAT,
                    [32 * SLICE_COUNT as u32, 32, 32],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        /*SimpleRenderPass::new_compute(
            rg.add_pass("csgi clear"),
            "/assets/shaders/csgi/clear_volume.hlsl",
        )
        .write(&mut cascade0)
        .dispatch(cascade0.desc().extent);*/

        SimpleRenderPass::new_rt(
            rg.add_pass("csgi trace"),
            "/assets/shaders/csgi/trace_volume.rgen.hlsl",
            &[
                "/assets/shaders/rt/triangle.rmiss.hlsl",
                "/assets/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/assets/shaders/rt/triangle.rchit.hlsl"],
        )
        .write(&mut cascade0)
        .constants(SLICE_DIRS)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(tlas, cascade0.desc().extent);

        let mut pretrace_hit_img = rg.create(
            ImageDesc::new_3d(
                vk::Format::R8_UNORM,
                [
                    PRETRACE_DIMS * PRETRACE_COUNT as u32,
                    PRETRACE_DIMS,
                    PRETRACE_DIMS,
                ],
            )
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
        );

        let mut pretrace_normal_img = rg.create(
            ImageDesc::new_3d(
                vk::Format::R8G8B8A8_UNORM,
                [
                    PRETRACE_DIMS * PRETRACE_COUNT as u32,
                    PRETRACE_DIMS,
                    PRETRACE_DIMS,
                ],
            )
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
        );

        let mut pretrace_col_img = rg.create(
            ImageDesc::new_3d(
                vk::Format::R16G16B16A16_SFLOAT,
                [
                    PRETRACE_DIMS * PRETRACE_COUNT as u32,
                    PRETRACE_DIMS,
                    PRETRACE_DIMS,
                ],
            )
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
        );

        SimpleRenderPass::new_compute(
            rg.add_pass("csgi clear"),
            "/assets/shaders/csgi/clear_volume.hlsl",
        )
        .write(&mut pretrace_hit_img)
        //.dispatch(pretrace_img.desc().div_up_extent([2, 2, 2]).extent);
        .dispatch(pretrace_col_img.desc().extent);

        SimpleRenderPass::new_rt(
            rg.add_pass("csgi pretrace"),
            "/assets/shaders/csgi/pretrace_volume.rgen.hlsl",
            &[
                "/assets/shaders/rt/triangle.rmiss.hlsl",
                "/assets/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/assets/shaders/rt/triangle.rchit.hlsl"],
        )
        .write(&mut pretrace_hit_img)
        .write(&mut pretrace_col_img)
        .write(&mut pretrace_normal_img)
        .read(&alt_cascade0)
        .constants((SLICE_DIRS, PRETRACE_DIRS))
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(
            tlas,
            [PRETRACE_DIMS * PRETRACE_COUNT as u32, PRETRACE_DIMS, 1],
        );

        let ray_dirs_canonical = [
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, -1.0),
            Vec3::new(0.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        ];

        let mut ray_dir_pretrace_indices = [[0u32; 4]; SLICE_COUNT * 9];
        for slice_idx in 0..SLICE_COUNT {
            let slice_dir = SLICE_DIRS[slice_idx];
            let slice_rot =
                build_orthonormal_basis(Vec3::new(slice_dir[0], slice_dir[1], slice_dir[2]));

            for dir_idx in 0..9 {
                let r_dir = slice_rot * ray_dirs_canonical[dir_idx].normalize();

                let mut highest_dot = -1.0;
                let mut pretraced_idx = 0;

                for i in 0..PRETRACE_COUNT {
                    let pretrace_dir = PRETRACE_DIRS[i];
                    let d = r_dir.dot(-Vec3::new(
                        pretrace_dir[0],
                        pretrace_dir[1],
                        pretrace_dir[2],
                    ));
                    if d > highest_dot {
                        highest_dot = d;
                        pretraced_idx = i;
                    }
                }

                ray_dir_pretrace_indices[slice_idx * 9 + dir_idx][0] = pretraced_idx as u32;
            }
        }

        SimpleRenderPass::new_compute(
            rg.add_pass("csgi sweep"),
            "/assets/shaders/csgi/sweep_volume.hlsl",
        )
        .write(&mut alt_cascade0)
        .read(&pretrace_hit_img)
        .read(&pretrace_col_img)
        .read(&pretrace_normal_img)
        .constants((SLICE_DIRS, PRETRACE_DIRS, ray_dir_pretrace_indices))
        .dispatch([32, 32, SLICE_COUNT as u32]);

        CsgiVolume {
            cascade0,
            alt_cascade0,
        }
    }
}

impl CsgiVolume {
    pub fn render_debug(
        &self,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer_depth: &GbufferDepth,
        out_img: &mut rg::Handle<Image>,
    ) {
        SimpleRenderPass::new_compute(
            rg.add_pass("csgi debug"),
            "/assets/shaders/csgi/render_debug.hlsl",
        )
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&self.cascade0)
        .read(&self.alt_cascade0)
        .write(out_img)
        .constants((out_img.desc().extent_inv_extent_2d(), SLICE_DIRS))
        .dispatch(out_img.desc().extent);
    }
}

fn build_orthonormal_basis(n: Vec3) -> Mat3 {
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

const SLICE_COUNT: usize = 16;
const SLICE_DIRS: [[f32; 4]; SLICE_COUNT] = [
    [0.85225946, 0.36958295, 0.37021905, 0.0],
    [0.263413, 0.9570777, 0.120897286, 0.0],
    [0.5356124, -0.75899905, -0.3701891, 0.0],
    [-0.97195244, 0.23506321, 0.0073164566, 0.0],
    [-0.544322, 0.38344887, 0.74611014, 0.0],
    [-0.6345936, -0.62937766, -0.448525, 0.0],
    [0.14410779, -0.35573938, -0.9234079, 0.0],
    [0.22534491, 0.5385581, -0.811896, 0.0],
    [-0.09385556, -0.98527056, 0.14294423, 0.0],
    [-0.7481385, -0.46306336, 0.47524846, 0.0],
    [0.08077474, -0.47626922, 0.8755816, 0.0],
    [-0.448539, 0.87202656, -0.1959147, 0.0],
    [0.8082192, -0.47252876, 0.3514234, 0.0],
    [0.22675002, 0.38531533, 0.8944922, 0.0],
    [0.87311435, 0.2334949, -0.4279624, 0.0],
    [-0.5878232, 0.17476612, -0.7898867, 0.0],
];

const PRETRACE_COUNT: usize = SLICE_COUNT;
const PRETRACE_DIRS: [[f32; 4]; PRETRACE_COUNT] = SLICE_DIRS;

/*const PRETRACE_COUNT: usize = 32;
const PRETRACE_DIRS: [[f32; 4]; PRETRACE_COUNT] = [
    [0.4949354, 0.65692055, 0.56876564, 0.0],
    [0.41396505, -0.5012211, -0.75987536, 0.0],
    [-0.32501304, -0.7347451, 0.5954126, 0.0],
    [-0.29773396, -0.16300744, 0.9406292, 0.0],
    [0.253012, 0.44385013, -0.8596405, 0.0],
    [-0.61467165, 0.68545514, 0.39029518, 0.0],
    [0.86047685, -0.25075623, -0.44351017, 0.0],
    [-0.84786266, 0.042965293, -0.5284728, 0.0],
    [0.070507474, -0.15328482, -0.9856637, 0.0],
    [0.6514735, -0.6411021, 0.40567276, 0.0],
    [-0.40091568, 0.6431644, -0.6523851, 0.0],
    [0.10629922, 0.90617836, -0.40931836, 0.0],
    [-0.53852654, 0.8311411, -0.13853982, 0.0],
    [-0.15250775, 0.6088128, 0.77851695, 0.0],
    [-0.022311788, -0.88675666, -0.461698, 0.0],
    [0.7510839, 0.29287472, -0.5916906, 0.0],
    [0.22023678, 0.17085586, 0.9603668, 0.0],
    [-0.4635353, -0.54665726, -0.6973528, 0.0],
    [-0.8111353, -0.3762356, 0.4477795, 0.0],
    [0.6084771, 0.7744858, -0.17299582, 0.0],
    [-0.96188545, 0.27330655, 0.0089388015, 0.0],
    [0.7185927, -0.00050469674, 0.6954308, 0.0],
    [-0.41999745, 0.11559472, -0.90013325, 0.0],
    [-0.89721835, -0.41877586, -0.14009362, 0.0],
    [0.90351343, 0.4026844, 0.14665882, 0.0],
    [0.111447796, -0.97055465, 0.21354994, 0.0],
    [0.97264284, -0.18890631, 0.13520618, 0.0],
    [-0.7308805, 0.17339776, 0.66011155, 0.0],
    [-0.4947147, -0.86822647, -0.03795531, 0.0],
    [0.04048167, 0.9674038, 0.24998304, 0.0],
    [0.23751135, -0.51609445, 0.82294285, 0.0],
    [0.5799199, -0.7884175, -0.20516169, 0.0],
];
*/
