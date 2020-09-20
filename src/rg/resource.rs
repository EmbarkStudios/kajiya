pub use crate::backend::image::{Image, ImageDesc};
use std::marker::PhantomData;

use super::resource_registry::AnyRenderResource;

//#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
//pub struct Image;

pub trait Resource {
    type Desc: ResourceDesc;
    type Impl;

    fn borrow_resource(res: &AnyRenderResource) -> &Self::Impl;
}

impl Resource for Image {
    type Desc = ImageDesc;
    type Impl = crate::backend::image::Image; // TODO: nuke

    fn borrow_resource(res: &AnyRenderResource) -> &Self::Impl {
        match res {
            AnyRenderResource::Image(img) => &**img,
            AnyRenderResource::Buffer(_) => unimplemented!(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum GraphResourceDesc {
    Image(ImageDesc),
}

impl From<ImageDesc> for GraphResourceDesc {
    fn from(desc: ImageDesc) -> Self {
        Self::Image(desc)
    }
}

pub trait ResourceDesc: Clone + std::fmt::Debug + Into<GraphResourceDesc> {
    type Resource: Resource;
}

impl ResourceDesc for ImageDesc {
    type Resource = Image;
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub(crate) struct GraphRawResourceHandle {
    pub(crate) id: u32,
    pub(crate) version: u32,
}

impl GraphRawResourceHandle {
    pub(crate) fn next_version(self) -> Self {
        Self {
            id: self.id,
            version: self.version + 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Handle<ResType: Resource> {
    pub(crate) raw: GraphRawResourceHandle,
    pub(crate) desc: <ResType as Resource>::Desc,
    pub(crate) marker: PhantomData<ResType>,
}

impl<ResType: Resource> PartialEq for Handle<ResType> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<ResType: Resource> Handle<ResType> {
    pub fn desc(&self) -> &<ResType as Resource>::Desc {
        &self.desc
    }
}

impl<ResType: Resource> Eq for Handle<ResType> {}

#[derive(Debug)]
pub struct Ref<ResType: Resource, ViewType: GpuViewType> {
    pub(crate) handle: GraphRawResourceHandle,
    pub(crate) desc: <ResType as Resource>::Desc,
    pub(crate) marker: PhantomData<(ResType, ViewType)>,
}

impl<ResType: Resource, ViewType: GpuViewType> Ref<ResType, ViewType> {
    pub fn desc(&self) -> &<ResType as Resource>::Desc {
        &self.desc
    }
}

impl<ResType: Resource, ViewType: GpuViewType> Clone for Ref<ResType, ViewType>
where
    <ResType as Resource>::Desc: Clone,
    ViewType: Clone,
{
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            desc: self.desc.clone(),
            marker: PhantomData,
        }
    }
}

impl<ResType: Resource, ViewType: GpuViewType> Copy for Ref<ResType, ViewType>
where
    <ResType as Resource>::Desc: Copy,
    ViewType: Copy,
{
}

/*impl<ResType: Resource, ViewType: GpuViewType> Ref<ResType, ViewType>
where
    <ResType as Resource>::Desc: Copy,
{
    pub(crate) fn internal_clone(&self) -> Ref<ResType, ViewType> {
        Ref {
            handle: self.handle,
            desc: self.desc,
            marker: PhantomData,
        }
    }
}*/

#[derive(Clone, Copy)]
pub struct GpuSrv;
pub struct GpuUav;
pub struct GpuRt;

pub trait GpuViewType {}
impl GpuViewType for GpuSrv {}
impl GpuViewType for GpuUav {}
impl GpuViewType for GpuRt {}

/*pub struct GpuResourceView<'a, ResType: Resource, ViewType: GpuViewType> {
    // TODO: not pub?
    pub res: &'a <ResType as Resource>::Impl,
    marker: PhantomData<ViewType>,
}

impl<'a, ResType: Resource, ViewType: GpuViewType> GpuResourceView<'a, ResType, ViewType> {
    pub fn new(res: &'a <ResType as Resource>::Impl) -> Self {
        Self {
            res,
            marker: PhantomData,
        }
    }
}
*/
