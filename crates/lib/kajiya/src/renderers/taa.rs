use super::PingPongTemporalResource;
use glam::Vec2;
use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg, SimpleRenderPass};

pub struct TaaRenderer {
    temporal_tex: PingPongTemporalResource,
    temporal_velocity_tex: PingPongTemporalResource,
    temporal_smooth_var_tex: PingPongTemporalResource,
    pub current_supersample_offset: Vec2,
}

impl Default for TaaRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl TaaRenderer {
    pub fn new() -> Self {
        Self {
            temporal_tex: PingPongTemporalResource::new("taa"),
            temporal_velocity_tex: PingPongTemporalResource::new("taa.velocity"),
            temporal_smooth_var_tex: PingPongTemporalResource::new("taa.smooth_var"),
            current_supersample_offset: Vec2::ZERO,
        }
    }
}

pub struct TaaOutput {
    pub temporal_out: rg::ReadOnlyHandle<Image>,
    pub this_frame_out: rg::Handle<Image>,
}

impl TaaRenderer {
    fn temporal_tex_desc(extent: [u32; 2]) -> ImageDesc {
        ImageDesc::new_2d(vk::Format::R16G16B16A16_SFLOAT, extent)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
    }

    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        input_tex: &rg::Handle<Image>,
        reprojection_map: &rg::Handle<Image>,
        depth_tex: &rg::Handle<Image>,
        output_extent: [u32; 2],
    ) -> TaaOutput {
        //let input_extent = input_tex.desc().extent_2d();

        let (mut temporal_output_tex, history_tex) = self
            .temporal_tex
            .get_output_and_history(rg, Self::temporal_tex_desc(output_extent));

        let (mut temporal_velocity_output_tex, velocity_history_tex) =
            self.temporal_velocity_tex.get_output_and_history(
                rg,
                ImageDesc::new_2d(vk::Format::R16G16_SFLOAT, output_extent)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            );

        let mut reprojected_history_img = rg.create(Self::temporal_tex_desc(output_extent));
        let mut closest_velocity_img =
            rg.create(ImageDesc::new_2d(vk::Format::R16G16_SFLOAT, output_extent));

        SimpleRenderPass::new_compute(
            rg.add_pass("reproject taa"),
            "/shaders/taa/reproject_history.hlsl",
        )
        .read(&history_tex)
        .read(reprojection_map)
        .read_aspect(depth_tex, vk::ImageAspectFlags::DEPTH)
        .write(&mut reprojected_history_img)
        .write(&mut closest_velocity_img)
        .constants((
            input_tex.desc().extent_inv_extent_2d(),
            reprojected_history_img.desc().extent_inv_extent_2d(),
        ))
        .dispatch(reprojected_history_img.desc().extent);

        let (mut smooth_var_output_tex, smooth_var_history_tex) =
            self.temporal_smooth_var_tex.get_output_and_history(
                rg,
                ImageDesc::new_2d(vk::Format::R16G16B16A16_SFLOAT, output_extent)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            );

        let mut filtered_input_img = rg.create(ImageDesc::new_2d(
            vk::Format::R16G16B16A16_SFLOAT,
            input_tex.desc().extent_2d(),
        ));

        let mut filtered_input_deviation_img = rg.create(ImageDesc::new_2d(
            vk::Format::R16G16B16A16_SFLOAT,
            input_tex.desc().extent_2d(),
        ));

        SimpleRenderPass::new_compute(
            rg.add_pass("taa filter input"),
            "/shaders/taa/filter_input.hlsl",
        )
        .read(input_tex)
        .read_aspect(depth_tex, vk::ImageAspectFlags::DEPTH)
        .write(&mut filtered_input_img)
        .write(&mut filtered_input_deviation_img)
        .dispatch(filtered_input_img.desc().extent);

        let mut filtered_history_img = rg.create(ImageDesc::new_2d(
            vk::Format::R16G16B16A16_SFLOAT,
            filtered_input_img.desc().extent_2d(),
        ));
        SimpleRenderPass::new_compute(
            rg.add_pass("taa filter history"),
            "/shaders/taa/filter_history.hlsl",
        )
        .read(&reprojected_history_img)
        .write(&mut filtered_history_img)
        .constants((
            reprojected_history_img.desc().extent_inv_extent_2d(),
            input_tex.desc().extent_inv_extent_2d(),
        ))
        .dispatch(filtered_history_img.desc().extent);

        let input_prob_img = {
            let mut input_prob_img = rg.create(ImageDesc::new_2d(
                vk::Format::R16_SFLOAT,
                input_tex.desc().extent_2d(),
            ));
            SimpleRenderPass::new_compute(
                rg.add_pass("taa input prob"),
                "/shaders/taa/input_prob.hlsl",
            )
            .read(input_tex)
            .read(&filtered_input_img)
            .read(&filtered_input_deviation_img)
            .read(&reprojected_history_img)
            .read(&filtered_history_img)
            .read(reprojection_map)
            .read_aspect(depth_tex, vk::ImageAspectFlags::DEPTH)
            .read(&smooth_var_history_tex)
            .read(&velocity_history_tex)
            .write(&mut input_prob_img)
            .constants((input_tex.desc().extent_inv_extent_2d(),))
            .dispatch(input_prob_img.desc().extent);

            let mut prob_filtered1_img = rg.create(*input_prob_img.desc());
            SimpleRenderPass::new_compute(
                rg.add_pass("taa prob filter"),
                "/shaders/taa/filter_prob.hlsl",
            )
            .read(&input_prob_img)
            .write(&mut prob_filtered1_img)
            .dispatch(prob_filtered1_img.desc().extent);

            let mut prob_filtered2_img = rg.create(*input_prob_img.desc());
            SimpleRenderPass::new_compute(
                rg.add_pass("taa prob filter2"),
                "/shaders/taa/filter_prob2.hlsl",
            )
            .read(&prob_filtered1_img)
            .write(&mut prob_filtered2_img)
            .dispatch(prob_filtered1_img.desc().extent);

            prob_filtered2_img
        };

        let mut this_frame_output_img = rg.create(Self::temporal_tex_desc(output_extent));
        SimpleRenderPass::new_compute(rg.add_pass("taa"), "/shaders/taa/taa.hlsl")
            .read(input_tex)
            .read(&reprojected_history_img)
            .read(reprojection_map)
            .read(&closest_velocity_img)
            .read(&velocity_history_tex)
            .read_aspect(depth_tex, vk::ImageAspectFlags::DEPTH)
            .read(&smooth_var_history_tex)
            .read(&input_prob_img)
            .write(&mut temporal_output_tex)
            .write(&mut this_frame_output_img)
            .write(&mut smooth_var_output_tex)
            .write(&mut temporal_velocity_output_tex)
            .constants((
                input_tex.desc().extent_inv_extent_2d(),
                temporal_output_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(temporal_output_tex.desc().extent);

        TaaOutput {
            temporal_out: temporal_output_tex.into(),
            this_frame_out: this_frame_output_img,
        }
    }
}
