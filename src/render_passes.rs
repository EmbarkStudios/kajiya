use crate::{
    render_client::{KajiyaRenderClient, RenderDebugMode},
    renderers::{
        deferred::light_gbuffer, post::post_process, raster_meshes::*,
        reference::reference_path_trace, shadows::trace_sun_shadow_mask, GbufferDepth,
    },
    vulkan::image::*,
    FrameState,
};
use kajiya_backend::{
    ash::vk::{self},
    vk_sync::{self, AccessType},
};
use kajiya_rg::{self as rg};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use rg::GetOrCreateTemporal;

impl KajiyaRenderClient {
    pub(super) fn prepare_render_graph_standard(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        frame_state: &FrameState,
    ) -> rg::ExportedHandle<Image> {
        let tlas = self.prepare_top_level_acceleration(rg);

        let mut accum_img = rg
            .get_or_create_temporal(
                "root.accum",
                ImageDesc::new_2d(
                    vk::Format::R32G32B32A32_SFLOAT,
                    [frame_state.window_cfg.width, frame_state.window_cfg.height],
                )
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_DST,
                ),
            )
            .unwrap();

        let sky_cube = crate::renderers::sky::render_sky_cube(rg);
        let convolved_sky_cube = crate::renderers::sky::convolve_cube(rg, &sky_cube);

        let csgi_volume = self.csgi.render(
            frame_state.camera_matrices.eye_position(),
            rg,
            &convolved_sky_cube,
            self.bindless_descriptor_set,
            &tlas,
        );

        let gbuffer_depth;
        let velocity_img;
        {
            let mut depth_img = rg.create(ImageDesc::new_2d(
                vk::Format::D24_UNORM_S8_UINT,
                frame_state.window_cfg.dims(),
            ));
            crate::renderers::imageops::clear_depth(rg, &mut depth_img);

            let mut gbuffer = rg.create(ImageDesc::new_2d(
                vk::Format::R32G32B32A32_SFLOAT,
                frame_state.window_cfg.dims(),
            ));
            crate::renderers::imageops::clear_color(rg, &mut gbuffer, [0.0, 0.0, 0.0, 0.0]);

            let mut mesh_velocity_img = rg.create(ImageDesc::new_2d(
                vk::Format::R16G16_SFLOAT,
                frame_state.window_cfg.dims(),
            ));

            if self.debug_mode != RenderDebugMode::CsgiVoxelGrid {
                raster_meshes(
                    rg,
                    self.raster_simple_render_pass.clone(),
                    &mut depth_img,
                    &mut gbuffer,
                    &mut mesh_velocity_img,
                    RasterMeshesData {
                        meshes: self.meshes.as_slice(),
                        instances: self.instances.as_slice(),
                        vertex_buffer: self.vertex_buffer.lock().clone(),
                        bindless_descriptor_set: self.bindless_descriptor_set,
                    },
                );
            }

            if self.debug_mode == RenderDebugMode::CsgiVoxelGrid {
                csgi_volume.debug_raster_voxel_grid(
                    rg,
                    self.raster_simple_render_pass.clone(),
                    &mut depth_img,
                    &mut gbuffer,
                    &mut mesh_velocity_img,
                );
            }

            gbuffer_depth = GbufferDepth::new(gbuffer, depth_img);
            velocity_img = mesh_velocity_img;
        }

        let reprojection_map = crate::renderers::reprojection::calculate_reprojection_map(
            rg,
            &gbuffer_depth.depth,
            &velocity_img,
        );

        let ssgi_tex = self
            .ssgi
            .render(rg, &gbuffer_depth, &reprojection_map, &accum_img);
        //let ssgi_tex = rg.create(ImageDesc::new_2d(vk::Format::R8_UNORM, [1, 1]));

        let sun_shadow_mask = trace_sun_shadow_mask(rg, &gbuffer_depth.depth, &tlas);

        let rtr = self.rtr.render(
            rg,
            &gbuffer_depth,
            &reprojection_map,
            &sky_cube,
            self.bindless_descriptor_set,
            &tlas,
            &csgi_volume,
        );

        let rtdgi = self.rtdgi.render(
            rg,
            &gbuffer_depth,
            &reprojection_map,
            &sky_cube,
            self.bindless_descriptor_set,
            &tlas,
            &csgi_volume,
            &ssgi_tex,
        );

        let mut debug_out_tex = rg.create(ImageDesc::new_2d(
            vk::Format::R16G16B16A16_SFLOAT,
            gbuffer_depth.gbuffer.desc().extent_2d(),
        ));

        light_gbuffer(
            rg,
            &gbuffer_depth.gbuffer,
            &gbuffer_depth.depth,
            &sun_shadow_mask,
            &ssgi_tex,
            &rtr,
            &rtdgi,
            &mut accum_img,
            &mut debug_out_tex,
            &csgi_volume,
            &sky_cube,
            self.bindless_descriptor_set,
        );

        let anti_aliased = self.taa.render(rg, &debug_out_tex, &reprojection_map);

        let mut post = post_process(rg, &anti_aliased, self.bindless_descriptor_set);

        self.render_ui(rg, &mut post);

        rg.export(
            post,
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        )
    }

    pub(super) fn prepare_render_graph_reference(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        frame_state: &FrameState,
    ) -> rg::ExportedHandle<Image> {
        let mut accum_img = rg
            .get_or_create_temporal(
                "refpt.accum",
                ImageDesc::new_2d(
                    vk::Format::R32G32B32A32_SFLOAT,
                    [frame_state.window_cfg.width, frame_state.window_cfg.height],
                )
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_DST,
                ),
            )
            .unwrap();

        if self.reset_reference_accumulation {
            self.reset_reference_accumulation = false;
            crate::renderers::imageops::clear_color(rg, &mut accum_img, [0.0, 0.0, 0.0, 0.0]);
        }

        let tlas = self.prepare_top_level_acceleration(rg);

        reference_path_trace(rg, &mut accum_img, self.bindless_descriptor_set, &tlas);

        let mut post = post_process(rg, &accum_img, self.bindless_descriptor_set);

        self.render_ui(rg, &mut post);

        rg.export(
            post,
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        )
    }

    fn render_ui(&mut self, rg: &mut rg::RenderGraph, target_img: &mut rg::Handle<Image>) {
        if let Some((ui_renderer, image)) = self.ui_frame.take() {
            let mut ui_tex = rg.import(image, AccessType::Nothing);
            let mut pass = rg.add_pass("render ui");

            pass.raster(&mut ui_tex, AccessType::ColorAttachmentWrite);
            pass.render(move |api| ui_renderer(api.cb.raw));

            rg::SimpleRenderPass::new_compute(
                rg.add_pass("blit ui"),
                "/assets/shaders/blit_ui.hlsl",
            )
            .read(&ui_tex)
            .write(target_img)
            .dispatch(target_img.desc().extent);
        }
    }
}
