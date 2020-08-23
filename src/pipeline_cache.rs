use crate::backend::shader::*;
use crate::shader_compiler::{CompileShader, CompiledShader};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use turbosloth::*;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct ComputePipelineHandle(usize);

struct ComputePipelineCacheEntry {
    lazy_handle: Lazy<CompiledShader>,
    desc: ComputePipelineDesc,
    pipeline: Option<Arc<ShaderPipeline>>,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RasterPipelineHandle(usize);

pub struct CompiledRasterShaders {
    shaders: Vec<RasterPipelineShader<Arc<CompiledShader>>>,
}

#[derive(Clone, Hash)]
pub struct CompileRasterShaders {
    shaders: Vec<RasterPipelineShader<PathBuf>>,
}

#[async_trait]
impl LazyWorker for CompileRasterShaders {
    type Output = anyhow::Result<CompiledRasterShaders>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let shaders = futures::future::try_join_all(self.shaders.iter().map(|shader| {
            CompileShader {
                path: shader.code.clone(),
                profile: match shader.desc.stage {
                    RasterStage::Vertex => "vs".to_owned(),
                    RasterStage::Pixel => "ps".to_owned(),
                },
            }
            .into_lazy()
            .eval(&ctx)
        }))
        .await?;

        let shaders = shaders
            .into_iter()
            .zip(self.shaders.iter())
            .map(|(shader, src_shader)| RasterPipelineShader {
                code: shader,
                desc: src_shader.desc.clone(),
            })
            .collect();

        Ok(CompiledRasterShaders { shaders })
    }
}

struct RasterPipelineCacheEntry {
    lazy_handle: Lazy<CompiledRasterShaders>,
    desc: RasterPipelineDesc,
    pipeline: Option<Arc<ShaderPipeline>>,
}

pub struct PipelineCache {
    lazy_cache: Arc<LazyCache>,
    compute_entries: HashMap<ComputePipelineHandle, ComputePipelineCacheEntry>,
    raster_entries: HashMap<RasterPipelineHandle, RasterPipelineCacheEntry>,
}

impl PipelineCache {
    pub fn new(lazy_cache: &Arc<LazyCache>) -> Self {
        Self {
            compute_entries: Default::default(),
            raster_entries: Default::default(),
            lazy_cache: lazy_cache.clone(),
        }
    }

    pub fn register_compute(
        &mut self,
        path: impl AsRef<Path>,
        desc: &ComputePipelineDescBuilder,
    ) -> ComputePipelineHandle {
        let handle = ComputePipelineHandle(self.compute_entries.len());
        self.compute_entries.insert(
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

    pub fn get_compute(&mut self, handle: ComputePipelineHandle) -> Arc<ShaderPipeline> {
        self.compute_entries
            .get(&handle)
            .unwrap()
            .pipeline
            .clone()
            .unwrap()
    }

    pub fn register_raster(
        &mut self,
        shaders: &[RasterPipelineShader<&str>],
        desc: &RasterPipelineDescBuilder,
    ) -> RasterPipelineHandle {
        let handle = RasterPipelineHandle(self.compute_entries.len());
        self.raster_entries.insert(
            handle,
            RasterPipelineCacheEntry {
                lazy_handle: CompileRasterShaders {
                    shaders: shaders
                        .iter()
                        .map(|shader| RasterPipelineShader {
                            code: PathBuf::from(shader.code),
                            desc: shader.desc.clone(),
                        })
                        .collect(),
                }
                .into_lazy(),
                desc: desc.clone().build().unwrap(),
                pipeline: None,
            },
        );
        handle
    }

    pub fn get_raster(&mut self, handle: RasterPipelineHandle) -> Arc<ShaderPipeline> {
        self.raster_entries
            .get(&handle)
            .unwrap()
            .pipeline
            .clone()
            .unwrap()
    }

    pub fn prepare_frame(
        &mut self,
        device: &Arc<crate::backend::device::Device>,
    ) -> anyhow::Result<()> {
        for entry in self.compute_entries.values_mut() {
            if entry.lazy_handle.is_stale() {
                // TODO: release
                entry.pipeline = None;
            }

            if entry.pipeline.is_none() {
                let compiled_shader = smol::block_on(entry.lazy_handle.eval(&self.lazy_cache))?;

                let pipeline =
                    create_compute_pipeline(&*device, &compiled_shader.spirv, &entry.desc);

                entry.pipeline = Some(Arc::new(pipeline));
            }
        }

        for entry in self.raster_entries.values_mut() {
            if entry.lazy_handle.is_stale() {
                // TODO: release
                entry.pipeline = None;
            }

            if entry.pipeline.is_none() {
                let compiled_shaders = smol::block_on(entry.lazy_handle.eval(&self.lazy_cache))?;

                let compiled_shaders = compiled_shaders
                    .shaders
                    .iter()
                    .map(|shader| RasterPipelineShader {
                        code: shader.code.spirv.as_slice(),
                        desc: shader.desc.clone(),
                    })
                    .collect::<Vec<_>>();

                let pipeline = create_raster_pipeline(&*device, &compiled_shaders, &entry.desc)?;

                entry.pipeline = Some(Arc::new(pipeline));
            }
        }

        Ok(())
    }
}
