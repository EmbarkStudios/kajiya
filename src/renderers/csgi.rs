// Cone sweep global illumination prototype

use rg::GetOrCreateTemporal;
use slingshot::{
    ash::vk,
    backend::{buffer::*, image::*, ray_tracing::RayTracingAcceleration},
    rg::{self, SimpleRenderPass},
    vk_sync, Device,
};

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
                    vk::Format::R32G32B32A32_SFLOAT,
                    [32 * SLICE_COUNT as u32, 32, 32],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        let mut alt_cascade0 = rg
            .get_or_create_temporal(
                "csgi.alt_cascade0",
                ImageDesc::new_3d(
                    vk::Format::R32G32B32A32_SFLOAT,
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

        let mut pretrace_img = rg.create(
            ImageDesc::new_3d(
                vk::Format::R32G32B32A32_SFLOAT,
                [32 * SLICE_COUNT as u32, 32, 32],
            )
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
        );

        SimpleRenderPass::new_compute(
            rg.add_pass("csgi clear"),
            "/assets/shaders/csgi/clear_volume.hlsl",
        )
        .write(&mut pretrace_img)
        .dispatch(pretrace_img.desc().extent);

        SimpleRenderPass::new_rt(
            rg.add_pass("csgi pretrace"),
            "/assets/shaders/csgi/pretrace_volume.rgen.hlsl",
            &[
                "/assets/shaders/rt/triangle.rmiss.hlsl",
                "/assets/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/assets/shaders/rt/triangle.rchit.hlsl"],
        )
        .write(&mut pretrace_img)
        .constants(PRETRACE_DIRS)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(tlas, [48 * PRETRACE_COUNT as u32, 48, 1]);

        SimpleRenderPass::new_compute(
            rg.add_pass("csgi sweep"),
            "/assets/shaders/csgi/sweep_volume.hlsl",
        )
        .write(&mut alt_cascade0)
        .read(&pretrace_img)
        .constants((SLICE_DIRS, PRETRACE_DIRS))
        .dispatch(pretrace_img.desc().extent);

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

const PRETRACE_COUNT: usize = 32;
const PRETRACE_DIRS: [[f32; 4]; PRETRACE_COUNT] = [
    [-0.2058445, -0.9775319, -0.045385186, 0.0],
    [0.6205791, 0.69949967, 0.35437566, 0.0],
    [-0.25882477, 0.21924709, -0.9407129, 0.0],
    [0.15851581, -0.74951863, 0.64272445, 0.0],
    [0.2923496, 0.926469, -0.23703794, 0.0],
    [-0.86993945, -0.3877663, 0.30470148, 0.0],
    [0.20771098, -0.1597723, 0.96505415, 0.0],
    [-0.6643184, -0.68040323, -0.30940714, 0.0],
    [-0.35620666, -0.30683753, 0.88259155, 0.0],
    [-0.978535, -0.07336217, -0.19258122, 0.0],
    [0.6947583, -0.44693312, 0.56352615, 0.0],
    [-0.69899225, 0.11056573, 0.7065305, 0.0],
    [-0.72199917, 0.50551444, -0.47241187, 0.0],
    [0.44339144, -0.8920058, 0.08792043, 0.0],
    [-0.20564492, -0.51824296, -0.8301414, 0.0],
    [-0.43421072, -0.77856106, 0.4531049, 0.0],
    [0.8789968, -0.46887758, -0.08671094, 0.0],
    [0.8442173, 0.506867, -0.1743079, 0.0],
    [-0.14838688, 0.7487025, -0.6460855, 0.0],
    [0.8680196, 0.00035104237, -0.49652994, 0.0],
    [-0.4798255, 0.66750276, 0.5693925, 0.0],
    [0.46549124, 0.5202455, -0.7160047, 0.0],
    [0.2817749, -0.017651062, -0.95931834, 0.0],
    [0.94956154, 0.06778596, 0.3061665, 0.0],
    [0.5504463, -0.4782836, -0.68429047, 0.0],
    [0.16556369, -0.8617419, -0.47957307, 0.0],
    [-0.42463315, 0.90376943, -0.053734668, 0.0],
    [0.07217759, 0.9302367, 0.35979146, 0.0],
    [0.57259715, 0.25155535, 0.7802902, 0.0],
    [-0.89515746, 0.417692, 0.15564874, 0.0],
    [-0.6836361, -0.14866261, -0.71452177, 0.0],
    [-0.0068429727, 0.47533596, 0.8797781, 0.0],
];
