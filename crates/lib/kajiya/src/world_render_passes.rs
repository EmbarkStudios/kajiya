use crate::{
    frame_desc::WorldFrameDesc,
    renderers::{
        deferred::light_gbuffer, motion_blur::motion_blur, post::post_process, raster_meshes::*,
        reference::reference_path_trace, shadows::trace_sun_shadow_mask, GbufferDepth,
    },
    world_renderer::{RenderDebugMode, WorldRenderer},
};
use kajiya_backend::{ash::vk, vk_sync, vulkan::image::*};
use kajiya_rg::{self as rg, GetOrCreateTemporal};

impl WorldRenderer {
    pub(super) fn prepare_render_graph_standard(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        frame_desc: &WorldFrameDesc,
    ) -> rg::ExportedHandle<Image> {
        let tlas = self.prepare_top_level_acceleration(rg);

        let mut accum_img = rg
            .get_or_create_temporal(
                "root.accum",
                ImageDesc::new_2d(vk::Format::R32G32B32A32_SFLOAT, frame_desc.render_extent).usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_DST,
                ),
            )
            .unwrap();

        let sky_cube = crate::renderers::sky::render_sky_cube(rg);
        let convolved_sky_cube = crate::renderers::sky::convolve_cube(rg, &sky_cube);

        let csgi_volume = self.csgi.render(
            frame_desc.camera_matrices.eye_position(),
            rg,
            &convolved_sky_cube,
            self.bindless_descriptor_set,
            &tlas,
        );

        let (gbuffer_depth, velocity_img) = {
            let mut gbuffer_depth = {
                let normal = rg.create(ImageDesc::new_2d(
                    vk::Format::A2R10G10B10_UNORM_PACK32,
                    frame_desc.render_extent,
                ));

                let gbuffer = rg.create(ImageDesc::new_2d(
                    vk::Format::R32G32B32A32_SFLOAT,
                    frame_desc.render_extent,
                ));

                let mut depth_img = rg.create(ImageDesc::new_2d(
                    vk::Format::D24_UNORM_S8_UINT,
                    frame_desc.render_extent,
                ));
                crate::renderers::imageops::clear_depth(rg, &mut depth_img);

                GbufferDepth::new(normal, gbuffer, depth_img)
            };

            let mut velocity_img = rg.create(ImageDesc::new_2d(
                vk::Format::R16G16B16A16_SFLOAT,
                frame_desc.render_extent,
            ));

            if self.debug_mode != RenderDebugMode::CsgiVoxelGrid {
                raster_meshes(
                    rg,
                    self.raster_simple_render_pass.clone(),
                    &mut gbuffer_depth,
                    &mut velocity_img,
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
                    &mut gbuffer_depth,
                    &mut velocity_img,
                );
            }

            (gbuffer_depth, velocity_img)
        };

        let reprojection_map = crate::renderers::reprojection::calculate_reprojection_map(
            rg,
            &gbuffer_depth,
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
            &gbuffer_depth,
            &sun_shadow_mask,
            &ssgi_tex,
            &rtr,
            &rtdgi,
            &mut accum_img,
            &mut debug_out_tex,
            &csgi_volume,
            &sky_cube,
            self.bindless_descriptor_set,
            self.debug_shading_mode,
        );

        let anti_aliased = self.taa.render(rg, &debug_out_tex, &reprojection_map);
        let motion_blurred =
            motion_blur(rg, &anti_aliased, &gbuffer_depth.depth, &reprojection_map);

        let post_processed = post_process(
            rg,
            &motion_blurred,
            self.bindless_descriptor_set,
            self.ev_shift,
        );

        rg.export(
            post_processed,
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        )
    }

    pub(super) fn prepare_render_graph_reference(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        frame_desc: &WorldFrameDesc,
    ) -> rg::ExportedHandle<Image> {
        let mut accum_img = rg
            .get_or_create_temporal(
                "refpt.accum",
                ImageDesc::new_2d(vk::Format::R32G32B32A32_SFLOAT, frame_desc.render_extent).usage(
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

        let post_processed =
            post_process(rg, &accum_img, self.bindless_descriptor_set, self.ev_shift);

        rg.export(
            post_processed,
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        )
    }
}
