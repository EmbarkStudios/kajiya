use kajiya_backend::vulkan::ray_tracing::RayTracingAcceleration;
pub use kajiya_backend::vulkan::{
    buffer::{Buffer, BufferDesc},
    image::*,
};
use std::marker::PhantomData;

use super::resource_registry::{AnyRenderResource, AnyRenderResourceRef};

pub trait Resource {
    type Desc: ResourceDesc;

    fn borrow_resource(res: &AnyRenderResource) -> &Self;
}

impl Resource for Image {
    type Desc = ImageDesc;

    fn borrow_resource(res: &AnyRenderResource) -> &Self {
        match res.borrow() {
            AnyRenderResourceRef::Image(img) => img,
            _ => unimplemented!(),
        }
    }
}

impl Resource for Buffer {
    type Desc = BufferDesc;

    fn borrow_resource(res: &AnyRenderResource) -> &Self {
        match res.borrow() {
            AnyRenderResourceRef::Buffer(buffer) => buffer,
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RayTracingAccelerationDesc;

impl Resource for RayTracingAcceleration {
    type Desc = RayTracingAccelerationDesc;

    fn borrow_resource(res: &AnyRenderResource) -> &Self {
        match res.borrow() {
            AnyRenderResourceRef::RayTracingAcceleration(inner) => inner,
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum GraphResourceDesc {
    Image(ImageDesc),
    Buffer(BufferDesc),
    RayTracingAcceleration(RayTracingAccelerationDesc),
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

impl From<RayTracingAccelerationDesc> for GraphResourceDesc {
    fn from(desc: RayTracingAccelerationDesc) -> Self {
        Self::RayTracingAcceleration(desc)
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

impl ResourceDesc for RayTracingAccelerationDesc {
    type Resource = RayTracingAcceleration;
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
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

#[derive(Debug)]
pub struct Handle<ResType: Resource> {
    pub(crate) raw: GraphRawResourceHandle,
    pub(crate) desc: <ResType as Resource>::Desc,
    pub(crate) marker: PhantomData<ResType>,
}

#[derive(Debug)]
pub struct ExportedHandle<ResType: Resource> {
    pub(crate) raw: GraphRawResourceHandle,
    pub(crate) marker: PhantomData<ResType>,
}

impl<ResType: Resource> Clone for ExportedHandle<ResType> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw,
            marker: PhantomData,
        }
    }
}

impl<ResType: Resource> Copy for ExportedHandle<ResType> {}

impl<ResType: Resource> PartialEq for Handle<ResType> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<ResType: Resource> Handle<ResType> {
    pub fn desc(&self) -> &<ResType as Resource>::Desc {
        &self.desc
    }

    pub(crate) fn clone_unchecked(&self) -> Self {
        Self {
            raw: self.raw,
            desc: self.desc.clone(),
            marker: PhantomData,
        }
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
            handle: self.handle,
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

pub trait GpuViewType {
    const IS_WRITABLE: bool;
}
impl GpuViewType for GpuSrv {
    const IS_WRITABLE: bool = false;
}
impl GpuViewType for GpuUav {
    const IS_WRITABLE: bool = true;
}
impl GpuViewType for GpuRt {
    const IS_WRITABLE: bool = true;
}
