use crate::backend::shader::*;
use crate::shader_compiler::{CompileShader, CompiledShader};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::{collections::HashMap, path::Path, sync::Arc};
use turbosloth::*;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct ComputePipelineHandle(usize);

struct ComputePipelineCacheEntry {
    lazy_handle: Lazy<CompiledShader>,
    ctor: Box<dyn Fn(&CompiledShader) -> ComputePipelineDescBuilder<'_, '_>>,
    pipeline: Option<Arc<ComputePipeline>>,
}

pub struct ComputePipelineCache {
    lazy_cache: Arc<LazyCache>,
    entries: HashMap<ComputePipelineHandle, ComputePipelineCacheEntry>,
}

impl ComputePipelineCache {
    pub fn new(lazy_cache: &Arc<LazyCache>) -> Self {
        Self {
            entries: Default::default(),
            lazy_cache: lazy_cache.clone(),
        }
    }

    pub fn register<Ctor>(&mut self, path: impl AsRef<Path>, ctor: Ctor) -> ComputePipelineHandle
    where
        Ctor: (Fn(&CompiledShader) -> ComputePipelineDescBuilder<'_, '_>) + 'static,
    {
        let handle = ComputePipelineHandle(self.entries.len());
        self.entries.insert(
            handle,
            ComputePipelineCacheEntry {
                lazy_handle: CompileShader {
                    path: path.as_ref().to_owned(),
                    profile: "cs".to_owned(),
                }
                .into_lazy(),
                ctor: Box::new(ctor),
                pipeline: None,
            },
        );
        handle
    }

    pub fn get(&mut self, handle: ComputePipelineHandle) -> Arc<ComputePipeline> {
        self.entries.get(&handle).unwrap().pipeline.clone().unwrap()
    }

    pub fn prepare_frame(
        &mut self,
        device: &Arc<crate::backend::device::Device>,
    ) -> anyhow::Result<()> {
        for entry in self.entries.values_mut() {
            if entry.lazy_handle.is_stale() {
                entry.pipeline = None;
            }

            if entry.pipeline.is_none() {
                let compiled_shader = smol::block_on(entry.lazy_handle.eval(&self.lazy_cache))?;

                let pipeline = create_compute_pipeline(
                    &*device,
                    (entry.ctor)(&*compiled_shader)
                        .entry_name("main")
                        .build()
                        .unwrap(),
                );

                entry.pipeline = Some(Arc::new(pipeline));
            }
        }

        Ok(())
    }
}

/*use crate::{
    backend::shader::ComputePipeline,
    shader_cache::{ShaderCache, ShaderCacheEntry, ShaderType},
};
use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, RwLock},
};

type CsToPipeline = HashMap<u64, Arc<ComputePipeline>>;

#[derive(Default)]
pub struct Pipelines {
    compute_shader_to_pipeline: CsToPipeline,
}

pub struct PipelineCache {
    pub shader_cache: ShaderCache,
    pub pipelines: Arc<RwLock<Pipelines>>,
}

impl PipelineCache {
    pub fn new(shader_cache: ShaderCache) -> Self {
        Self {
            shader_cache: shader_cache,
            pipelines: Default::default(),
        }
    }

    pub fn get_or_load_compute(&self, path: &Path) -> anyhow::Result<Arc<ComputePipeline>> {
        let shader_cache_entry = self.shader_cache.get_or_load(ShaderType::Compute, path);

        let mut pipelines = self.pipelines.write().unwrap();
        let compute_pipes = &mut pipelines.compute_shader_to_pipeline;

        if let Some(retired) = shader_cache_entry.retired {
            compute_pipes.remove(&retired.identity());
        }

        let shader = shader_cache_entry.entry?;

        Ok(match compute_pipes.entry(shader.identity()) {
            std::collections::hash_map::Entry::Occupied(occupied) => occupied.get().clone(),
            std::collections::hash_map::Entry::Vacant(vacant) => {
                let shader = match &*shader {
                    ShaderCacheEntry::Compute(shader) => shader,
                    ShaderCacheEntry::Raster(..) => unreachable!(),
                };

                let shader_handle = shader.shader_handle;

                let pipeline_handle = params
                    .handles
                    .allocate_persistent(RenderResourceType::ComputePipelineState);

                params.device.create_compute_pipeline_state(
                    pipeline_handle,
                    &RenderComputePipelineStateDesc {
                        shader: shader_handle,
                        shader_signature: RenderShaderSignatureDesc::new(
                            &[RenderShaderParameter::new(
                                shader.srvs.len() as u32,
                                shader.uavs.len() as u32,
                            )],
                            &[],
                        ),
                    },
                    "compute pipeline".into(),
                )?;

                let pipeline_entry = Arc::new(ComputePipeline {
                    handle: pipeline_handle,
                    group_size: shader.group_size,
                    srvs: shader.srvs.clone(),
                    uavs: shader.uavs.clone(),
                });

                vacant.insert(pipeline_entry.clone());
                pipeline_entry
            }
        })
    }
}
*/
