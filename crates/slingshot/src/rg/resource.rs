pub use crate::backend::{
    buffer::{Buffer, BufferDesc},
    image::{Image, ImageDesc, ImageViewDescBuilder},
};
use std::marker::PhantomData;

use super::resource_registry::{AnyRenderResource, AnyRenderResourceRef};

pub trait Resource {
    type Desc: ResourceDesc;
    type Impl;

    fn borrow_resource(res: &AnyRenderResource) -> &Self::Impl;
}

impl Resource for Image {
    type Desc = ImageDesc;
    type Impl = crate::backend::image::Image; // TODO: nuke

    fn borrow_resource(res: &AnyRenderResource) -> &Self::Impl {
        match res.borrow() {
            AnyRenderResourceRef::Image(img) => img,
            AnyRenderResourceRef::Buffer(_) => unimplemented!(),
        }
    }
}

impl Resource for Buffer {
    type Desc = BufferDesc;
    type Impl = crate::backend::buffer::Buffer; // TODO: nuke

    fn borrow_resource(res: &AnyRenderResource) -> &Self::Impl {
        match res.borrow() {
            AnyRenderResourceRef::Image(_) => unimplemented!(),
            AnyRenderResourceRef::Buffer(buffer) => buffer,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum GraphResourceDesc {
    Image(ImageDesc),
    Buffer(BufferDesc),
}

impl From<ImageDesc> for GraphResourceDesc {
    fn from(desc: ImageDesc) -> Self {
        Self::Image(desc)
    }
}

impl From<BufferDesc> for GraphResourceDesc {
    fn from(desc: BufferDesc) -> Self {
        Self::Buffer(desc)
    }
}

pub trait ResourceDesc: Clone + std::fmt::Debug + Into<GraphResourceDesc> {
    type Resource: Resource;
}

impl ResourceDesc for ImageDesc {
    type Resource = Image;
}

impl ResourceDesc for BufferDesc {
    type Resource = Buffer;
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

#[derive(Clone, Debug)]
pub struct ExportedHandle<ResType: Resource>(pub(crate) Handle<ResType>);

impl<ResType: Resource> PartialEq for Handle<ResType> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<ResType: Resource> Handle<ResType> {
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

#[derive(Clone, Copy)]
pub struct GpuSrv;
pub struct GpuUav;
pub struct GpuRt;

pub trait GpuViewType {}
impl GpuViewType for GpuSrv {}
impl GpuViewType for GpuUav {}
impl GpuViewType for GpuRt {}
