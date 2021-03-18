use crate::shader_compiler::{CompileShader, CompiledShader};
use crate::vulkan::{
    ray_tracing::{create_ray_tracing_pipeline, RayTracingPipeline, RayTracingPipelineDesc},
    shader::*,
};
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
    pipeline: Option<Arc<ComputePipeline>>,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RasterPipelineHandle(usize);

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RtPipelineHandle(usize);

pub struct CompiledPipelineShaders {
    shaders: Vec<PipelineShader<Arc<CompiledShader>>>,
}

#[derive(Clone, Hash)]
pub struct CompilePipelineShaders {
    shaders: Vec<PipelineShader<PathBuf>>,
}

#[async_trait]
impl LazyWorker for CompilePipelineShaders {
    type Output = anyhow::Result<CompiledPipelineShaders>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let shaders = futures::future::try_join_all(self.shaders.iter().map(|shader| {
            CompileShader {
                path: shader.code.clone(),
                profile: match shader.desc.stage {
                    ShaderPipelineStage::Vertex => "vs".to_owned(),
                    ShaderPipelineStage::Pixel => "ps".to_owned(),
                    ShaderPipelineStage::RayGen
                    | ShaderPipelineStage::RayMiss
                    | ShaderPipelineStage::RayClosestHit => "lib".to_owned(),
                },
            }
            .into_lazy()
            .eval(&ctx)
        }))
        .await?;

        let shaders = shaders
            .into_iter()
            .zip(self.shaders.iter())
            .map(|(shader, src_shader)| PipelineShader {
                code: shader,
                desc: src_shader.desc.clone(),
            })
            .collect();

        Ok(CompiledPipelineShaders { shaders })
    }
}

struct RasterPipelineCacheEntry {
    lazy_handle: Lazy<CompiledPipelineShaders>,
    desc: RasterPipelineDesc,
    pipeline: Option<Arc<RasterPipeline>>,
}

struct RtPipelineCacheEntry {
    lazy_handle: Lazy<CompiledPipelineShaders>,
    desc: RayTracingPipelineDesc,
    pipeline: Option<Arc<RayTracingPipeline>>,
}

pub struct PipelineCache {
    lazy_cache: Arc<LazyCache>,

    compute_entries: HashMap<ComputePipelineHandle, ComputePipelineCacheEntry>,
    raster_entries: HashMap<RasterPipelineHandle, RasterPipelineCacheEntry>,
    rt_entries: HashMap<RtPipelineHandle, RtPipelineCacheEntry>,

    path_to_handle: HashMap<PathBuf, ComputePipelineHandle>,

    raster_shaders_to_handle: HashMap<Vec<PipelineShader<&'static str>>, RasterPipelineHandle>,
    rt_shaders_to_handle: HashMap<Vec<PipelineShader<&'static str>>, RtPipelineHandle>,
}

impl PipelineCache {
    pub fn new(lazy_cache: &Arc<LazyCache>) -> Self {
        Self {
            lazy_cache: lazy_cache.clone(),

            compute_entries: Default::default(),
            raster_entries: Default::default(),
            rt_entries: Default::default(),

            path_to_handle: Default::default(),

            raster_shaders_to_handle: Default::default(),
            rt_shaders_to_handle: Default::default(),
        }
    }

    // TODO: should probably use the `desc` as key as well
    pub fn register_compute(
        &mut self,
        path: impl AsRef<Path>,
        desc: &ComputePipelineDesc,
    ) -> ComputePipelineHandle {
        match self.path_to_handle.entry(path.as_ref().to_owned()) {
            std::collections::hash_map::Entry::Occupied(occupied) => *occupied.get(),
            std::collections::hash_map::Entry::Vacant(vacant) => {
                let handle = ComputePipelineHandle(self.compute_entries.len());
                self.compute_entries.insert(
                    handle,
                    ComputePipelineCacheEntry {
                        lazy_handle: CompileShader {
                            path: path.as_ref().to_owned(),
                            profile: "cs".to_owned(),
                        }
                        .into_lazy(),
                        desc: desc.clone(),
                        pipeline: None,
                    },
                );
                vacant.insert(handle);
                handle
            }
        }
    }

    pub fn get_compute(&self, handle: ComputePipelineHandle) -> Arc<ComputePipeline> {
        self.compute_entries
            .get(&handle)
            .unwrap()
            .pipeline
            .clone()
            .unwrap()
    }

    pub fn register_raster(
        &mut self,
        shaders: &[PipelineShader<&'static str>],
        desc: &RasterPipelineDesc,
    ) -> RasterPipelineHandle {
        if let Some(handle) = self.raster_shaders_to_handle.get(shaders) {
            return *handle;
        }

        let handle = RasterPipelineHandle(self.raster_entries.len());
        self.raster_shaders_to_handle
            .insert(shaders.to_owned(), handle);
        self.raster_entries.insert(
            handle,
            RasterPipelineCacheEntry {
                lazy_handle: CompilePipelineShaders {
                    shaders: shaders
                        .iter()
                        .map(|shader| PipelineShader {
                            code: PathBuf::from(shader.code),
                            desc: shader.desc.clone(),
                        })
                        .collect(),
                }
                .into_lazy(),
                desc: desc.clone(),
                pipeline: None,
            },
        );
        handle
    }

    pub fn get_raster(&self, handle: RasterPipelineHandle) -> Arc<RasterPipeline> {
        self.raster_entries
            .get(&handle)
            .unwrap()
            .pipeline
            .clone()
            .unwrap()
    }

    pub fn register_ray_tracing(
        &mut self,
        shaders: &[PipelineShader<&'static str>],
        desc: &RayTracingPipelineDesc,
    ) -> RtPipelineHandle {
        if let Some(handle) = self.rt_shaders_to_handle.get(shaders) {
            return *handle;
        }

        let handle = RtPipelineHandle(self.rt_entries.len());
        self.rt_shaders_to_handle.insert(shaders.to_owned(), handle);
        self.rt_entries.insert(
            handle,
            RtPipelineCacheEntry {
                lazy_handle: CompilePipelineShaders {
                    shaders: shaders
                        .iter()
                        .map(|shader| PipelineShader {
                            code: PathBuf::from(shader.code),
                            desc: shader.desc.clone(),
                        })
                        .collect(),
                }
                .into_lazy(),
                desc: desc.clone(),
                pipeline: None,
            },
        );
        handle
    }

    pub fn get_ray_tracing(&self, handle: RtPipelineHandle) -> Arc<RayTracingPipeline> {
        self.rt_entries
            .get(&handle)
            .unwrap()
            .pipeline
            .clone()
            .unwrap()
    }

    fn invalidate_stale_pipelines(&mut self) {
        for entry in self.compute_entries.values_mut() {
            if entry.pipeline.is_some() && entry.lazy_handle.is_stale() {
                // TODO: release
                entry.pipeline = None;
            }
        }

        for entry in self.raster_entries.values_mut() {
            if entry.pipeline.is_some() && entry.lazy_handle.is_stale() {
                // TODO: release
                entry.pipeline = None;
            }
        }

        for entry in self.rt_entries.values_mut() {
            if entry.pipeline.is_some() && entry.lazy_handle.is_stale() {
                // TODO: release
                entry.pipeline = None;
            }
        }
    }

    // TODO: create pipelines right away too
    pub fn parallel_compile_shaders(&mut self) -> anyhow::Result<()> {
        let compute = self.compute_entries.values().filter_map(|entry| {
            entry.pipeline.is_none().then(|| {
                let task = entry.lazy_handle.eval(&self.lazy_cache);
                smol::spawn(async move { task.await.map(|_| ()) })
            })
        });

        let raster = self.raster_entries.values().filter_map(|entry| {
            entry.pipeline.is_none().then(|| {
                let task = entry.lazy_handle.eval(&self.lazy_cache);
                smol::spawn(async move { task.await.map(|_| ()) })
            })
        });

        let rt = self.rt_entries.values().filter_map(|entry| {
            entry.pipeline.is_none().then(|| {
                let task = entry.lazy_handle.eval(&self.lazy_cache);
                smol::spawn(async move { task.await.map(|_| ()) })
            })
        });

        let shaders: Vec<_> = compute.chain(raster).chain(rt).collect();

        if !shaders.is_empty() {
            let _ = smol::block_on(futures::future::try_join_all(shaders))?;
        }

        Ok(())
    }

    pub fn prepare_frame(
        &mut self,
        device: &Arc<crate::vulkan::device::Device>,
    ) -> anyhow::Result<()> {
        self.invalidate_stale_pipelines();
        self.parallel_compile_shaders()?;

        for entry in self.compute_entries.values_mut() {
            if entry.pipeline.is_none() {
                let compiled_shader = smol::block_on(entry.lazy_handle.eval(&self.lazy_cache))?;

                let pipeline =
                    create_compute_pipeline(&*device, &compiled_shader.spirv, &entry.desc);

                entry.pipeline = Some(Arc::new(pipeline));
            }
        }

        for entry in self.raster_entries.values_mut() {
            if entry.pipeline.is_none() {
                let compiled_shaders = smol::block_on(entry.lazy_handle.eval(&self.lazy_cache))?;
                assert!(!entry.lazy_handle.is_stale());

                let compiled_shaders = compiled_shaders
                    .shaders
                    .iter()
                    .map(|shader| PipelineShader {
                        code: shader.code.spirv.as_slice(),
                        desc: shader.desc.clone(),
                    })
                    .collect::<Vec<_>>();

                let pipeline = create_raster_pipeline(&*device, &compiled_shaders, &entry.desc)?;

                entry.pipeline = Some(Arc::new(pipeline));
            }
        }

        for entry in self.rt_entries.values_mut() {
            if entry.pipeline.is_none() {
                let compiled_shaders = smol::block_on(entry.lazy_handle.eval(&self.lazy_cache))?;

                let compiled_shaders = compiled_shaders
                    .shaders
                    .iter()
                    .map(|shader| PipelineShader {
                        code: shader.code.spirv.as_slice(),
                        desc: shader.desc.clone(),
                    })
                    .collect::<Vec<_>>();

                let pipeline =
                    create_ray_tracing_pipeline(&*device, &compiled_shaders, &entry.desc)?;

                entry.pipeline = Some(Arc::new(pipeline));
            }
        }

        Ok(())
    }
}
