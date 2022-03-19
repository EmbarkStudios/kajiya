use crate::{
    frame_desc::WorldFrameDesc,
    renderers::{
        deferred::light_gbuffer, motion_blur::motion_blur, post::post_process, raster_meshes::*,
        reference::reference_path_trace, shadows::trace_sun_shadow_mask, GbufferDepth,
    },
    world_renderer::{RenderDebugMode, WorldRenderer},
};
use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg, GetOrCreateTemporal};

impl WorldRenderer {
    pub(super) fn prepare_render_graph_standard(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        frame_desc: &WorldFrameDesc,
    ) -> rg::Handle<Image> {
        let tlas = self.prepare_top_level_acceleration(rg);

        let mut accum_img = rg
            .get_or_create_temporal(
                "root.accum",
                ImageDesc::new_2d(vk::Format::R16G16B16A16_SFLOAT, frame_desc.render_extent).usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_DST,
                ),
            )
            .unwrap();

        let sky_cube = crate::renderers::sky::render_sky_cube(rg);
        let convolved_sky_cube = crate::renderers::sky::convolve_cube(rg, &sky_cube);

        let (mut gbuffer_depth, velocity_img) = {
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
                    vk::Format::D32_SFLOAT,
                    frame_desc.render_extent,
                ));
                rg::imageops::clear_depth(rg, &mut depth_img);

                GbufferDepth::new(normal, gbuffer, depth_img)
            };

            let mut velocity_img = rg.create(ImageDesc::new_2d(
                vk::Format::R16G16B16A16_SFLOAT,
                frame_desc.render_extent,
            ));

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

            (gbuffer_depth, velocity_img)
        };

        let mut surfel_state = self.surfel_gi.allocate_surfels(rg, &mut gbuffer_depth);

        let wrc = crate::renderers::wrc::wrc_trace(
            rg,
            &mut surfel_state,
            &sky_cube,
            self.bindless_descriptor_set,
            &tlas,
        );

        surfel_state.trace_irradiance(rg, &sky_cube, self.bindless_descriptor_set, &tlas, &wrc);

        let reprojection_map = crate::renderers::reprojection::calculate_reprojection_map(
            rg,
            &gbuffer_depth,
            &velocity_img,
        );

        let ssgi_tex = self.ssgi.render(
            rg,
            &gbuffer_depth,
            &reprojection_map,
            &accum_img,
            self.bindless_descriptor_set,
        );
        //let ssgi_tex = rg.create(ImageDesc::new_2d(vk::Format::R8_UNORM, [1, 1]));

        let sun_shadow_mask =
            trace_sun_shadow_mask(rg, &gbuffer_depth, &tlas, self.bindless_descriptor_set);

        let denoised_shadow_mask = if self.sun_size_multiplier > 0.0f32 {
            self.shadow_denoise
                .render(rg, &gbuffer_depth, &sun_shadow_mask, &reprojection_map)
        } else {
            sun_shadow_mask.into()
        };

        let rtdgi = self.rtdgi.render(
            rg,
            &gbuffer_depth,
            &reprojection_map,
            &sky_cube,
            self.bindless_descriptor_set,
            &mut surfel_state,
            &wrc,
            &tlas,
            &ssgi_tex,
        );

        // TODO: don't iter over all the things
        let any_triangle_lights = self
            .instances
            .iter()
            .any(|inst| !self.mesh_lights[inst.mesh.0].lights.is_empty());

        let mut rtr = self.rtr.trace(
            rg,
            &gbuffer_depth,
            &reprojection_map,
            &sky_cube,
            self.bindless_descriptor_set,
            &tlas,
            &rtdgi,
            &mut surfel_state,
            &wrc,
        );

        if any_triangle_lights {
            // Render specular lighting into the RTR image so they can be jointly filtered
            self.lighting.render_specular(
                &mut rtr.resolved_tex,
                rg,
                &gbuffer_depth,
                self.bindless_descriptor_set,
                &tlas,
            );
        }

        let rtr = rtr.filter_temporal(rg, &gbuffer_depth, &reprojection_map);

        let mut debug_out_tex = rg.create(ImageDesc::new_2d(
            vk::Format::R16G16B16A16_SFLOAT,
            gbuffer_depth.gbuffer.desc().extent_2d(),
        ));

        light_gbuffer(
            rg,
            &gbuffer_depth,
            &denoised_shadow_mask,
            &ssgi_tex,
            &rtr,
            &rtdgi,
            &mut surfel_state,
            &wrc,
            &mut accum_img,
            &mut debug_out_tex,
            &sky_cube,
            &convolved_sky_cube,
            self.bindless_descriptor_set,
            self.debug_shading_mode,
            self.debug_show_wrc,
        );

        #[allow(unused_mut)]
        let mut anti_aliased = None;

        #[cfg(feature = "dlss")]
        if self.use_dlss {
            anti_aliased = Some(self.dlss.render(
                rg,
                &debug_out_tex,
                &reprojection_map,
                &gbuffer_depth.depth,
                self.temporal_upscale_extent,
            ));
        }

        //let dof = crate::renderers::dof::dof(rg, &debug_out_tex, &gbuffer_depth.depth);

        let anti_aliased = anti_aliased.unwrap_or_else(|| {
            self.taa
                .render(
                    rg,
                    //&dof,
                    &debug_out_tex,
                    &reprojection_map,
                    &gbuffer_depth.depth,
                    self.temporal_upscale_extent,
                )
                .this_frame_out
        });

        let mut final_post_input =
            motion_blur(rg, &anti_aliased, &gbuffer_depth.depth, &reprojection_map);

        if matches!(self.debug_mode, RenderDebugMode::WorldRadianceCache) {
            wrc.see_through(
                rg,
                &sky_cube,
                &mut surfel_state,
                self.bindless_descriptor_set,
                &tlas,
                &mut final_post_input,
            );
        }

        let post_processed = post_process(
            rg,
            &final_post_input,
            //&anti_aliased,
            self.bindless_descriptor_set,
            self.ev_shift,
        );

        rg.debugged_resource.take().unwrap_or(post_processed)
    }

    pub(super) fn prepare_render_graph_reference(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        frame_desc: &WorldFrameDesc,
    ) -> rg::Handle<Image> {
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
            rg::imageops::clear_color(rg, &mut accum_img, [0.0, 0.0, 0.0, 0.0]);
        }

        let tlas = self.prepare_top_level_acceleration(rg);

        reference_path_trace(rg, &mut accum_img, self.bindless_descriptor_set, &tlas);

        post_process(
            rg,
            &accum_img,
            //&accum_img, // hack
            self.bindless_descriptor_set,
            self.ev_shift,
        )
    }
}
