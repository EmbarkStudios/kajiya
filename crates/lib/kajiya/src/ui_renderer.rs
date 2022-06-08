// Used to support Dear Imgui, though could be used for other immediate-mode rendering too

use std::sync::Arc;

use kajiya_backend::{ash::vk, vk_sync::AccessType, vulkan::image::*, BackendError};
use kajiya_rg::{self as rg};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

#[derive(Default)]
pub struct UiRenderer {
    pub ui_frame: Option<(UiRenderCallback, Arc<Image>)>,
}

pub type UiRenderCallback =
    Box<dyn (FnOnce(vk::CommandBuffer) -> Result<(), BackendError>) + 'static>;

impl UiRenderer {
    pub fn prepare_render_graph(&mut self, rg: &mut rg::TemporalRenderGraph) -> rg::Handle<Image> {
        self.render_ui(rg)
    }

    fn render_ui(&mut self, rg: &mut rg::RenderGraph) -> rg::Handle<Image> {
        if let Some((ui_renderer, image)) = self.ui_frame.take() {
            let mut ui_tex = rg.import(image, AccessType::Nothing);
            let mut pass = rg.add_pass("ui");

            pass.raster(&mut ui_tex, AccessType::ColorAttachmentWrite);
            pass.render(move |api| ui_renderer(api.cb.raw));

            ui_tex
        } else {
            let mut blank_img = rg.create(ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, [1, 1]));
            rg::imageops::clear_color(rg, &mut blank_img, [0.0f32; 4]);
            blank_img
        }
    }
}
