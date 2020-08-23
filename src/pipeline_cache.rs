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
    desc: ComputePipelineDesc,
    pipeline: Option<Arc<ShaderPipeline>>,
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

    pub fn register(
        &mut self,
        path: impl AsRef<Path>,
        desc: &ComputePipelineDescBuilder,
    ) -> ComputePipelineHandle {
        let handle = ComputePipelineHandle(self.entries.len());
        self.entries.insert(
            handle,
            ComputePipelineCacheEntry {
                lazy_handle: CompileShader {
                    path: path.as_ref().to_owned(),
                    profile: "cs".to_owned(),
                }
                .into_lazy(),
                desc: desc.clone().build().unwrap(),
                pipeline: None,
            },
        );
        handle
    }

    pub fn get(&mut self, handle: ComputePipelineHandle) -> Arc<ShaderPipeline> {
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

                let pipeline =
                    create_compute_pipeline(&*device, &compiled_shader.spirv, "main", &entry.desc);

                entry.pipeline = Some(Arc::new(pipeline));
            }
        }

        Ok(())
    }
}
