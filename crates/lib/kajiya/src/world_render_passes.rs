use crate::{
    frame_desc::WorldFrameDesc,
    renderers::{
        deferred::light_gbuffer, motion_blur::motion_blur, raster_meshes::*,
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
        let tlas = if rg.device().ray_tracing_enabled() {
            Some(self.prepare_top_level_acceleration(rg))
        } else {
            None
        };

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

        let mut ircache_state = self.ircache.prepare(rg);

        let wrc = /*if let Some(tlas) = tlas.as_ref() {
            crate::renderers::wrc::wrc_trace(
                rg,
                &mut ircache_state,
                &sky_cube,
                self.bindless_descriptor_set,
                tlas,
            )
        } else */{
            crate::renderers::wrc::allocate_dummy_output(rg)
        };

        let traced_ircache = tlas.as_ref().map(|tlas| {
            ircache_state.trace_irradiance(rg, &sky_cube, self.bindless_descriptor_set, tlas, &wrc)
        });

        let sun_shadow_mask = if let Some(tlas) = tlas.as_ref() {
            trace_sun_shadow_mask(rg, &gbuffer_depth, tlas, self.bindless_descriptor_set)
        } else {
            rg.create(gbuffer_depth.depth.desc().format(vk::Format::R8_UNORM))
        };

        let reprojected_rtdgi = self.rtdgi.reproject(rg, &reprojection_map);

        let denoised_shadow_mask = if self.sun_size_multiplier > 0.0f32 {
            self.shadow_denoise
                .render(rg, &gbuffer_depth, &sun_shadow_mask, &reprojection_map)
        } else {
            sun_shadow_mask.into()
        };

        if let Some(traced_ircache) = traced_ircache {
            ircache_state.sum_up_irradiance_for_sampling(rg, traced_ircache);
        }

        let rtdgi = if let Some(tlas) = tlas.as_ref() {
            self.rtdgi.render(
                rg,
                reprojected_rtdgi,
                &gbuffer_depth,
                &reprojection_map,
                &sky_cube,
                self.bindless_descriptor_set,
                &mut ircache_state,
                &wrc,
                &tlas,
                &ssgi_tex,
            )
        } else {
            rg.create(ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, [1, 1]))
                .into()
        };

        // TODO: don't iter over all the things
        let any_triangle_lights = self
            .instances
            .iter()
            .any(|inst| !self.mesh_lights[inst.mesh.0].lights.is_empty());

        let mut rtr = if let Some(tlas) = tlas.as_ref() {
            self.rtr.trace(
                rg,
                &gbuffer_depth,
                &reprojection_map,
                &sky_cube,
                self.bindless_descriptor_set,
                &tlas,
                &rtdgi,
                &mut ircache_state,
                &wrc,
            )
        } else {
            self.rtr.create_dummy_output(rg, &gbuffer_depth)
        };

        if any_triangle_lights {
            if let Some(tlas) = tlas.as_ref() {
                // Render specular lighting into the RTR image so they can be jointly filtered
                self.lighting.render_specular(
                    &mut rtr.resolved_tex,
                    rg,
                    &gbuffer_depth,
                    self.bindless_descriptor_set,
                    tlas,
                );
            }
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
            &mut ircache_state,
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

        if let Some(tlas) = tlas.as_ref() {
            if matches!(self.debug_mode, RenderDebugMode::WorldRadianceCache) {
                wrc.see_through(
                    rg,
                    &sky_cube,
                    &mut ircache_state,
                    self.bindless_descriptor_set,
                    tlas,
                    &mut final_post_input,
                );
            }
        }

        let post_processed = self.post.render(
            rg,
            &final_post_input,
            //&anti_aliased,
            self.bindless_descriptor_set,
            self.exposure_state().post_mult,
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

        if rg.device().ray_tracing_enabled() {
            let tlas = self.prepare_top_level_acceleration(rg);

            reference_path_trace(rg, &mut accum_img, self.bindless_descriptor_set, &tlas);
        }

        self.post.render(
            rg,
            &accum_img,
            //&accum_img, // hack
            self.bindless_descriptor_set,
            self.exposure_state().post_mult,
        )
    }
}
