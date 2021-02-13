use std::cell::{Ref, RefCell};

use slingshot::{rg, Image};

pub mod csgi;
pub mod half_res;
pub mod rtr;
pub mod ssgi;
pub mod surfel_gi;

pub struct GbufferDepth {
    pub gbuffer: rg::Handle<Image>,
    pub depth: rg::Handle<Image>,
    half_view_normal: RefCell<Option<rg::Handle<Image>>>,
    half_depth: RefCell<Option<rg::Handle<Image>>>,
}

impl GbufferDepth {
    pub fn new(gbuffer: rg::Handle<Image>, depth: rg::Handle<Image>) -> Self {
        Self {
            gbuffer,
            depth,
            half_view_normal: Default::default(),
            half_depth: Default::default(),
        }
    }

    pub fn half_view_normal(&self, rg: &mut rg::RenderGraph) -> Ref<rg::Handle<Image>> {
        if self.half_view_normal.borrow().is_none() {
            *self.half_view_normal.borrow_mut() = Some(
                half_res::extract_half_res_gbuffer_view_normal_rgba8(rg, &self.gbuffer),
            );
        }

        Ref::map(self.half_view_normal.borrow(), |res| res.as_ref().unwrap())
    }

    pub fn half_depth(&self, rg: &mut rg::RenderGraph) -> Ref<rg::Handle<Image>> {
        if self.half_depth.borrow().is_none() {
            *self.half_depth.borrow_mut() = Some(half_res::extract_half_res_depth(rg, &self.depth));
        }

        Ref::map(self.half_depth.borrow(), |res| res.as_ref().unwrap())
    }
}
