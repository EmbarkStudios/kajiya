// Cone sweep global illumination prototype

use std::sync::Arc;

use glam::Vec3;
use kajiya_backend::{
    ash::{version::DeviceV1_0, vk},
    vk_sync::AccessType,
    vulkan::{
        image::*,
        ray_tracing::RayTracingAcceleration,
        shader::{
            PipelineShader, PipelineShaderDesc, RasterPipelineDesc, RenderPass, ShaderPipelineStage,
        },
    },
};
use kajiya_rg::{
    self as rg, BindRgRef, GetOrCreateTemporal, IntoRenderPassPipelineBinding, SimpleRenderPass,
};

use super::GbufferDepth;

const VOLUME_DIMS: u32 = 64;

pub struct CsgiRenderer {
    pub trace_subdiv: i32,
    pub neighbors_per_frame: i32,
}

impl Default for CsgiRenderer {
    fn default() -> Self {
        Self {
            trace_subdiv: 3,
            neighbors_per_frame: 2,
        }
    }
}

pub struct CsgiVolume {
    pub direct_cascade0: rg::Handle<Image>,
    pub indirect_cascade0: rg::Handle<Image>,
}

impl CsgiRenderer {
    pub fn render(
        &mut self,
        _eye_position: Vec3,
        rg: &mut rg::TemporalRenderGraph,
        sky_cube: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
    ) -> CsgiVolume {
        let mut direct_cascade0 = rg
            .get_or_create_temporal(
                "csgi.direct_cascade0",
                ImageDesc::new_3d(
                    //vk::Format::B10G11R11_UFLOAT_PACK32,
                    vk::Format::R16G16B16A16_SFLOAT,
                    [
                        VOLUME_DIMS * PRETRACE_COUNT as u32,
                        VOLUME_DIMS,
                        VOLUME_DIMS,
                    ],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        let mut indirect_cascade0 = rg
            .get_or_create_temporal(
                "csgi.indirect_cascade0",
                ImageDesc::new_3d(
                    //vk::Format::B10G11R11_UFLOAT_PACK32,
                    vk::Format::R16G16B16A16_SFLOAT,
                    [
                        VOLUME_DIMS * TRACE_COUNT as u32,
                        VOLUME_DIMS * 4,
                        VOLUME_DIMS,
                    ],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        let mut indirect_cascade_combined0 = rg
            .get_or_create_temporal(
                "csgi.indirect_cascade_combined0",
                ImageDesc::new_3d(
                    //vk::Format::B10G11R11_UFLOAT_PACK32,
                    vk::Format::R16G16B16A16_SFLOAT,
                    [VOLUME_DIMS * TRACE_COUNT as u32, VOLUME_DIMS, VOLUME_DIMS],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        /*SimpleRenderPass::new_compute(
            rg.add_pass("csgi clear"),
            "/shaders/csgi/clear_volume.hlsl",
        )
        .write(&mut direct_cascade0)
        .dispatch(direct_cascade0.desc().extent);*/

        let sweep_vx_count = VOLUME_DIMS >> self.trace_subdiv.clamp(0, 5);

        SimpleRenderPass::new_rt(
            rg.add_pass("csgi trace"),
            "/shaders/csgi/trace_volume.rgen.hlsl",
            &[
                "/shaders/rt/gbuffer.rmiss.hlsl",
                "/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/shaders/rt/gbuffer.rchit.hlsl"],
        )
        .read(&indirect_cascade_combined0)
        .write(&mut direct_cascade0)
        .constants(sweep_vx_count)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(
            tlas,
            [
                VOLUME_DIMS * PRETRACE_COUNT as u32,
                VOLUME_DIMS as u32,
                VOLUME_DIMS / sweep_vx_count,
            ],
        );

        SimpleRenderPass::new_compute(rg.add_pass("csgi sweep"), "/shaders/csgi/sweep_volume.hlsl")
            .read(&direct_cascade0)
            .read(sky_cube)
            .write(&mut indirect_cascade0)
            .dispatch([VOLUME_DIMS, VOLUME_DIMS, PRETRACE_COUNT as u32]);

        SimpleRenderPass::new_compute(
            rg.add_pass("csgi diagonal sweep"),
            "/shaders/csgi/diagonal_sweep_volume.hlsl",
        )
        .read(&direct_cascade0)
        .read(sky_cube)
        .write(&mut indirect_cascade0)
        .dispatch([
            VOLUME_DIMS,
            VOLUME_DIMS,
            (TRACE_COUNT - PRETRACE_COUNT) as u32,
        ]);

        SimpleRenderPass::new_compute(
            rg.add_pass("csgi subray combine"),
            "/shaders/csgi/subray_combine.hlsl",
        )
        .read(&indirect_cascade0)
        .read(&direct_cascade0)
        .write(&mut indirect_cascade_combined0)
        .dispatch([VOLUME_DIMS * (TRACE_COUNT as u32), VOLUME_DIMS, VOLUME_DIMS]);

        CsgiVolume {
            direct_cascade0,
            indirect_cascade0: indirect_cascade_combined0,
        }
    }
}

impl CsgiVolume {
    pub fn debug_raster_voxel_grid(
        &self,
        rg: &mut rg::RenderGraph,
        render_pass: Arc<RenderPass>,
        gbuffer_depth: &mut GbufferDepth,
        velocity_img: &mut rg::Handle<Image>,
    ) {
        let mut pass = rg.add_pass("raster csgi voxels");

        let pipeline = pass.register_raster_pipeline(
            &[
                PipelineShader {
                    code: "/shaders/csgi/raster_voxels_vs.hlsl",
                    desc: PipelineShaderDesc::builder(ShaderPipelineStage::Vertex)
                        .build()
                        .unwrap(),
                },
                PipelineShader {
                    code: "/shaders/csgi/raster_voxels_ps.hlsl",
                    desc: PipelineShaderDesc::builder(ShaderPipelineStage::Pixel)
                        .build()
                        .unwrap(),
                },
            ],
            RasterPipelineDesc::builder()
                .render_pass(render_pass.clone())
                .face_cull(true),
        );

        let depth_ref = pass.raster(
            &mut gbuffer_depth.depth,
            AccessType::DepthStencilAttachmentWrite,
        );

        let geometric_normal_ref = pass.raster(
            &mut gbuffer_depth.geometric_normal,
            AccessType::ColorAttachmentWrite,
        );
        let gbuffer_ref = pass.raster(&mut gbuffer_depth.gbuffer, AccessType::ColorAttachmentWrite);
        let velocity_ref = pass.raster(velocity_img, AccessType::ColorAttachmentWrite);

        let grid_ref = pass.read(
            &self.direct_cascade0,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        pass.render(move |api| {
            let [width, height, _] = gbuffer_ref.desc().extent;

            api.begin_render_pass(
                &*render_pass,
                [width, height],
                &[
                    (gbuffer_ref, &ImageViewDesc::default()),
                    (velocity_ref, &ImageViewDesc::default()),
                    (geometric_normal_ref, &ImageViewDesc::default()),
                ],
                Some((
                    depth_ref,
                    &ImageViewDesc::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                        .build()
                        .unwrap(),
                )),
            );

            api.set_default_view_and_scissor([width, height]);

            let _ = api.bind_raster_pipeline(
                pipeline
                    .into_binding()
                    .descriptor_set(0, &[grid_ref.bind()]),
            );

            unsafe {
                let raw_device = &api.device().raw;
                let cb = api.cb;

                raw_device.cmd_draw(
                    cb.raw,
                    // 6 verts (two triangles) per cube face
                    6 * PRETRACE_COUNT as u32 * VOLUME_DIMS * VOLUME_DIMS * VOLUME_DIMS,
                    1,
                    0,
                    0,
                );
            }

            api.end_render_pass();
        });
    }
}

const PRETRACE_COUNT: usize = 6;
const TRACE_COUNT: usize = 6 + 8;
