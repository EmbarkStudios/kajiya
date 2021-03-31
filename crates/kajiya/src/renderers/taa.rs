use super::PingPongTemporalResource;
use glam::Vec2;
use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg, SimpleRenderPass};

pub struct TaaRenderer {
    temporal_tex: PingPongTemporalResource,
    pub current_supersample_offset: Vec2,
}

impl TaaRenderer {
    pub fn new() -> Self {
        Self {
            temporal_tex: PingPongTemporalResource::new("taa"),
            current_supersample_offset: Vec2::zero(),
        }
    }
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
    ) -> rg::ReadOnlyHandle<Image> {
        let (mut temporal_output_tex, history_tex) = self
            .temporal_tex
            .get_output_and_history(rg, Self::temporal_tex_desc(input_tex.desc().extent_2d()));

        SimpleRenderPass::new_compute(rg.add_pass("taa"), "/assets/shaders/taa/taa.hlsl")
            .read(&input_tex)
            .read(&history_tex)
            .read(reprojection_map)
            .write(&mut temporal_output_tex)
            .constants((
                input_tex.desc().extent_inv_extent_2d(),
                self.current_supersample_offset,
            ))
            .dispatch(input_tex.desc().extent);

        temporal_output_tex.into()
    }
}
