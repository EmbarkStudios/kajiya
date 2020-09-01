use crate::backend::image::ImageDesc;
use std::marker::PhantomData;

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Image;

pub trait Resource {
    type Desc: ResourceDesc;
}

impl Resource for Image {
    type Desc = ImageDesc;
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
pub struct Ref<ResType: Resource, AccessMode> {
    pub(crate) handle: GraphRawResourceHandle,
    pub(crate) desc: <ResType as Resource>::Desc,
    pub(crate) marker: PhantomData<(ResType, AccessMode)>,
}

impl<ResType: Resource, AccessMode> Ref<ResType, AccessMode> {
    pub fn desc(&self) -> &<ResType as Resource>::Desc {
        &self.desc
    }
}

impl<ResType: Resource, AccessMode> Clone for Ref<ResType, AccessMode>
where
    <ResType as Resource>::Desc: Clone,
    AccessMode: Clone,
{
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            desc: self.desc.clone(),
            marker: PhantomData,
        }
    }
}

impl<ResType: Resource, AccessMode> Copy for Ref<ResType, AccessMode>
where
    <ResType as Resource>::Desc: Copy,
    AccessMode: Copy,
{
}

impl<ResType: Resource, AccessMode> Ref<ResType, AccessMode>
where
    <ResType as Resource>::Desc: Copy,
{
    pub(crate) fn internal_clone(&self) -> Ref<ResType, AccessMode> {
        Ref {
            handle: self.handle,
            desc: self.desc,
            marker: PhantomData,
        }
    }
}

#[derive(Clone, Copy)]
pub struct GpuSrv;
pub struct GpuUav;
pub struct GpuRt;

pub struct GpuResourceView<ViewType, ResType> {
    res: ResType,
    marker: PhantomData<ViewType>,
}

pub trait ToGpuResourceView {
    type ResType;

    fn to_gpu_resource_view(res: Self::ResType) -> Self;
}

impl<ResType> ToGpuResourceView for GpuResourceView<GpuSrv, ResType> {
    type ResType = ResType;

    fn to_gpu_resource_view(res: Self::ResType) -> Self {
        Self {
            res,
            marker: PhantomData,
        }
    }
}

impl<ResType> ToGpuResourceView for GpuResourceView<GpuUav, ResType> {
    type ResType = ResType;

    fn to_gpu_resource_view(res: Self::ResType) -> Self {
        Self {
            res,
            marker: PhantomData,
        }
    }
}

impl<ResType> ToGpuResourceView for GpuResourceView<GpuRt, ResType> {
    type ResType = ResType;

    fn to_gpu_resource_view(res: Self::ResType) -> Self {
        Self {
            res,
            marker: PhantomData,
        }
    }
}
