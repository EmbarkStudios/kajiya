use std::{
    collections::{hash_map, HashMap},
    sync::Arc,
};

use vk_sync::AccessType;

use crate::{Device, Image, ImageDesc};

use super::{
    Buffer, BufferDesc, ExportableGraphResource, ExportedHandle, Handle, RenderGraph, Resource,
    ResourceDesc, RetiredRenderGraph, TypeEquals,
};

pub struct ReadOnlyHandle<ResType: Resource>(Handle<ResType>);

impl<ResType: Resource> std::ops::Deref for ReadOnlyHandle<ResType> {
    type Target = Handle<ResType>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<ResType: Resource> From<Handle<ResType>> for ReadOnlyHandle<ResType> {
    fn from(h: Handle<ResType>) -> Self {
        Self(h)
    }
}

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct TemporalResourceKey(String);

impl<'a> From<&'a str> for TemporalResourceKey {
    fn from(s: &'a str) -> Self {
        TemporalResourceKey(String::from(s))
    }
}

#[derive(Clone)]
enum TemporalResource {
    Image(Arc<Image>),
    Buffer(Arc<Buffer>),
}

enum ExportedResourceHandle {
    Image(ExportedHandle<Image>),
    Buffer(ExportedHandle<Buffer>),
}

enum TemporalResourceState {
    Inert {
        resource: TemporalResource,
        access_type: vk_sync::AccessType,
    },
    Imported {
        resource: TemporalResource,
        handle: ExportableGraphResource,
    },
    Exported {
        resource: TemporalResource,
        handle: ExportedResourceHandle,
    },
}

#[derive(Default)]
pub struct TemporalRenderGraphState {
    resources: HashMap<TemporalResourceKey, TemporalResourceState>,
}

impl TemporalRenderGraphState {
    pub(crate) fn clone_assuming_inert(&self) -> Self {
        Self {
            resources: self
                .resources
                .iter()
                .map(|(k, v)| match v {
                    TemporalResourceState::Inert {
                        resource,
                        access_type,
                    } => (
                        k.clone(),
                        TemporalResourceState::Inert {
                            resource: resource.clone(),
                            access_type: *access_type,
                        },
                    ),
                    TemporalResourceState::Imported { .. }
                    | TemporalResourceState::Exported { .. } => {
                        panic!("Not in inert state!")
                    }
                })
                .collect(),
        }
    }
}

pub struct ExportedTemporalRenderGraphState(TemporalRenderGraphState);

pub struct TemporalRenderGraph {
    rg: RenderGraph,
    device: Arc<Device>,
    temporal_state: TemporalRenderGraphState,
}

impl std::ops::Deref for TemporalRenderGraph {
    type Target = RenderGraph;

    fn deref(&self) -> &Self::Target {
        &self.rg
    }
}

impl std::ops::DerefMut for TemporalRenderGraph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.rg
    }
}

impl TemporalRenderGraph {
    pub fn new(state: TemporalRenderGraphState, device: Arc<Device>) -> Self {
        Self {
            rg: RenderGraph::new(),
            device,
            temporal_state: state,
        }
    }
}

pub trait GetOrCreateTemporal<Desc: ResourceDesc> {
    fn get_or_create_temporal(
        &mut self,
        key: impl Into<TemporalResourceKey>,
        desc: Desc,
        //) -> anyhow::Result<Handle<Image>> {
    ) -> anyhow::Result<Handle<<Desc as ResourceDesc>::Resource>>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>;
}

impl GetOrCreateTemporal<ImageDesc> for TemporalRenderGraph {
    fn get_or_create_temporal(
        &mut self,
        key: impl Into<TemporalResourceKey>,
        desc: ImageDesc,
        //) -> anyhow::Result<Handle<Image>> {
    ) -> anyhow::Result<Handle<Image>> {
        let key = key.into();

        match self.temporal_state.resources.entry(key.clone()) {
            hash_map::Entry::Occupied(mut entry) => {
                let state = entry.get_mut();

                match state {
                    TemporalResourceState::Inert {
                        resource,
                        access_type,
                    } => {
                        let resource = resource.clone();

                        match &resource {
                            TemporalResource::Image(image) => {
                                let handle = self.rg.import(image.clone(), *access_type);

                                *state = TemporalResourceState::Imported {
                                    resource,
                                    handle: ExportableGraphResource::Image(
                                        handle.clone_unchecked(),
                                    ),
                                };

                                Ok(handle)
                            }
                            TemporalResource::Buffer(_) => {
                                anyhow::bail!(
                                    "Resource {:?} is a buffer, but an image was requested",
                                    key
                                );
                            }
                        }
                    }
                    TemporalResourceState::Imported { .. } => Err(anyhow::anyhow!(
                        "Temporal resource already taken: {:?}",
                        key
                    )),
                    TemporalResourceState::Exported { .. } => {
                        unreachable!()
                    }
                }
            }
            hash_map::Entry::Vacant(entry) => {
                let resource = Arc::new(self.device.create_image(desc, None)?);
                let handle = self.rg.import(resource.clone(), AccessType::Nothing);
                entry.insert(TemporalResourceState::Imported {
                    resource: TemporalResource::Image(resource),
                    handle: ExportableGraphResource::Image(handle.clone_unchecked()),
                });
                Ok(handle)
            }
        }
    }
}

impl GetOrCreateTemporal<BufferDesc> for TemporalRenderGraph {
    fn get_or_create_temporal(
        &mut self,
        key: impl Into<TemporalResourceKey>,
        desc: BufferDesc,
        //) -> anyhow::Result<Handle<Image>> {
    ) -> anyhow::Result<Handle<Buffer>> {
        let key = key.into();

        match self.temporal_state.resources.entry(key.clone()) {
            hash_map::Entry::Occupied(mut entry) => {
                let state = entry.get_mut();

                match state {
                    TemporalResourceState::Inert {
                        resource,
                        access_type,
                    } => {
                        let resource = resource.clone();

                        match &resource {
                            TemporalResource::Buffer(buffer) => {
                                let handle = self.rg.import(buffer.clone(), *access_type);

                                *state = TemporalResourceState::Imported {
                                    resource,
                                    handle: ExportableGraphResource::Buffer(
                                        handle.clone_unchecked(),
                                    ),
                                };

                                Ok(handle)
                            }
                            TemporalResource::Image(_) => {
                                anyhow::bail!(
                                    "Resource {:?} is an image, but a buffer was requested",
                                    key
                                );
                            }
                        }
                    }
                    TemporalResourceState::Imported { .. } => Err(anyhow::anyhow!(
                        "Temporal resource already taken: {:?}",
                        key
                    )),
                    TemporalResourceState::Exported { .. } => {
                        unreachable!()
                    }
                }
            }
            hash_map::Entry::Vacant(entry) => {
                let resource = Arc::new(self.device.create_buffer(desc, None)?);
                let handle = self.rg.import(resource.clone(), AccessType::Nothing);
                entry.insert(TemporalResourceState::Imported {
                    resource: TemporalResource::Buffer(resource),
                    handle: ExportableGraphResource::Buffer(handle.clone_unchecked()),
                });
                Ok(handle)
            }
        }
    }
}

impl TemporalRenderGraph {
    pub fn export_temporal(self) -> (RenderGraph, ExportedTemporalRenderGraphState) {
        let mut rg = self.rg;
        let mut state = self.temporal_state;

        for (_, state) in &mut state.resources {
            match state {
                TemporalResourceState::Inert { .. } => {
                    // Nothing to do here
                }
                TemporalResourceState::Imported { resource, handle } => match handle {
                    ExportableGraphResource::Image(handle) => {
                        let handle = rg.export(handle.clone_unchecked(), AccessType::Nothing);
                        *state = TemporalResourceState::Exported {
                            resource: resource.clone(),
                            handle: ExportedResourceHandle::Image(handle),
                        }
                    }
                    ExportableGraphResource::Buffer(handle) => {
                        let handle = rg.export(handle.clone_unchecked(), AccessType::Nothing);
                        *state = TemporalResourceState::Exported {
                            resource: resource.clone(),
                            handle: ExportedResourceHandle::Buffer(handle),
                        }
                    }
                },
                TemporalResourceState::Exported { .. } => {
                    unreachable!()
                }
            }
        }

        (rg, ExportedTemporalRenderGraphState(state))
    }
}

impl ExportedTemporalRenderGraphState {
    pub fn retire_temporal(self, rg: &RetiredRenderGraph) -> TemporalRenderGraphState {
        let mut state = self.0;

        for (_, state) in &mut state.resources {
            match state {
                TemporalResourceState::Inert { .. } => {
                    // Nothing to do here
                }
                TemporalResourceState::Imported { .. } => {
                    unreachable!()
                }
                TemporalResourceState::Exported { resource, handle } => match handle {
                    ExportedResourceHandle::Image(handle) => {
                        *state = TemporalResourceState::Inert {
                            resource: resource.clone(),
                            access_type: rg.exported_resource(*handle).1,
                        }
                    }
                    ExportedResourceHandle::Buffer(handle) => {
                        *state = TemporalResourceState::Inert {
                            resource: resource.clone(),
                            access_type: rg.exported_resource(*handle).1,
                        }
                    }
                },
            }
        }

        state
    }
}
