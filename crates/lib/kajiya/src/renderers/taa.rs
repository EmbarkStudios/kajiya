use super::PingPongTemporalResource;
use glam::Vec2;
use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg, SimpleRenderPass};

pub struct TaaRenderer {
    temporal_tex: PingPongTemporalResource,
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
            current_supersample_offset: Vec2::ZERO,
        }
    }
}

pub struct TaaOutput {
    pub color: rg::ReadOnlyHandle<Image>,
    pub debug: rg::Handle<Image>,
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
        let (mut temporal_output_tex, history_tex) = self
            .temporal_tex
            .get_output_and_history(rg, Self::temporal_tex_desc(output_extent));

        let mut reprojected_history_img = rg.create(Self::temporal_tex_desc(output_extent));
        SimpleRenderPass::new_compute(
            rg.add_pass("reproject taa"),
            "/shaders/taa/reproject_history.hlsl",
        )
        .read(&history_tex)
        .read(reprojection_map)
        .read_aspect(depth_tex, vk::ImageAspectFlags::DEPTH)
        .write(&mut reprojected_history_img)
        .constants((
            input_tex.desc().extent_inv_extent_2d(),
            reprojected_history_img.desc().extent_inv_extent_2d(),
        ))
        .dispatch(reprojected_history_img.desc().extent);

        let mut debug_output_img = rg.create(Self::temporal_tex_desc(output_extent));
        SimpleRenderPass::new_compute(rg.add_pass("taa"), "/shaders/taa/taa.hlsl")
            .read(input_tex)
            .read(&reprojected_history_img)
            .read(reprojection_map)
            .read_aspect(depth_tex, vk::ImageAspectFlags::DEPTH)
            .write(&mut temporal_output_tex)
            .write(&mut debug_output_img)
            .constants((
                input_tex.desc().extent_inv_extent_2d(),
                temporal_output_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(temporal_output_tex.desc().extent);

        TaaOutput {
            color: temporal_output_tex.into(),
            debug: debug_output_img,
        }
    }
}
