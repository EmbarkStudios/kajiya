use crate::shader_compiler::CompiledShader;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use turbosloth::*;

pub struct ComputeShaderCacheEntry {
    pub compiled_shader: Arc<CompiledShader>,
    pub identity: u64,
    //pub group_size: [u32; 3], // TODO
}

pub struct RasterShaderCacheEntry {
    pub compiled_shader: Arc<CompiledShader>,
    pub stage: ShaderType,
    pub identity: u64,
}

// TODO: figure out the ownership model -- should this release the resources?
pub enum ShaderCacheEntry {
    Compute(ComputeShaderCacheEntry),
    Raster(RasterShaderCacheEntry),
}

impl ShaderCacheEntry {
    pub fn compiled_shader(&self) -> &CompiledShader {
        match self {
            Self::Compute(ComputeShaderCacheEntry {
                compiled_shader, ..
            })
            | Self::Raster(RasterShaderCacheEntry {
                compiled_shader, ..
            }) => compiled_shader,
        }
    }

    pub fn identity(&self) -> u64 {
        match self {
            Self::Compute(ComputeShaderCacheEntry { identity, .. })
            | Self::Raster(RasterShaderCacheEntry { identity, .. }) => *identity,
        }
    }
}

/*impl std::hash::Hash for ShaderCacheEntry {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identity().hash(state)
    }
}
impl PartialEq for ShaderCacheEntry {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}
impl Eq for ShaderCacheEntry {}*/

pub struct ShaderCacheOutput {
    pub entry: anyhow::Result<Arc<ShaderCacheEntry>>,
    pub retired: Option<Arc<ShaderCacheEntry>>,
}

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub enum ShaderType {
    Vertex,
    Pixel,
    Compute,
}

impl ShaderType {
    fn as_profile(self) -> &'static str {
        match self {
            Self::Vertex => "vs",
            Self::Pixel => "ps",
            Self::Compute => "cs",
        }
    }
}

#[derive(Hash, PartialEq, Eq)]
struct ShaderCacheKey {
    path: PathBuf,
    shader_type: ShaderType,
}

struct TurboslothShaderCacheEntry {
    lazy_handle: OpaqueLazy,
    entry: Arc<ShaderCacheEntry>,
}

pub struct TurboslothShaderCache {
    shaders: RwLock<HashMap<ShaderCacheKey, TurboslothShaderCacheEntry>>,
    lazy_cache: Arc<LazyCache>,
}

impl TurboslothShaderCache {
    pub fn new(lazy_cache: Arc<LazyCache>) -> Self {
        Self {
            shaders: Default::default(),
            lazy_cache,
        }
    }
}

impl TurboslothShaderCache {
    fn compile_shader(
        &self,
        shader_type: ShaderType,
        path: &Path,
    ) -> anyhow::Result<TurboslothShaderCacheEntry> {
        let path = path;

        match shader_type {
            ShaderType::Vertex | ShaderType::Pixel => {
                let lazy_shader = crate::shader_compiler::CompileShader {
                    path: path.to_owned(),
                    profile: shader_type.as_profile().to_owned(),
                }
                .into_lazy();

                let compiled_shader = smol::block_on(lazy_shader.eval(&self.lazy_cache))?;

                Ok(TurboslothShaderCacheEntry {
                    entry: Arc::new(ShaderCacheEntry::Raster(RasterShaderCacheEntry {
                        compiled_shader,
                        stage: shader_type,
                        identity: lazy_shader.identity(),
                    })),
                    lazy_handle: lazy_shader.into_opaque(),
                })
            }
            ShaderType::Compute => {
                let lazy_shader = crate::shader_compiler::CompileShader {
                    path: path.to_owned(),
                    profile: "cs".to_owned(),
                }
                .into_lazy();

                let compiled_shader = smol::block_on(lazy_shader.eval(&self.lazy_cache))?;

                Ok(TurboslothShaderCacheEntry {
                    entry: Arc::new(ShaderCacheEntry::Compute(ComputeShaderCacheEntry {
                        compiled_shader,
                        identity: lazy_shader.identity(),
                        //group_size: shader_data.group_size,
                    })),
                    lazy_handle: lazy_shader.into_opaque(),
                })
            }
        }
    }

    fn get_or_load_impl(
        &self,
        shader_type: ShaderType,
        path: &Path,
        retired: &mut Option<Arc<ShaderCacheEntry>>,
    ) -> anyhow::Result<Arc<ShaderCacheEntry>> {
        let key = ShaderCacheKey {
            path: path.to_owned(),
            shader_type,
        };

        let mut shaders = self.shaders.write().unwrap();

        // If the shader's lazy handle is stale, force re-compilation
        if let Some(entry) = shaders.get(&key) {
            if entry.lazy_handle.is_up_to_date() {
                return Ok(entry.entry.clone());
            } else {
                *retired = shaders.remove(&key).map(|entry| entry.entry);
            }
        }

        let new_entry = self.compile_shader(shader_type, path)?;
        let result = new_entry.entry.clone();
        shaders.insert(key, new_entry);
        Ok(result)
    }
}

impl TurboslothShaderCache {
    pub fn get_or_load(&self, shader_type: ShaderType, path: &Path) -> ShaderCacheOutput {
        let mut retired = None;
        let entry = self.get_or_load_impl(shader_type, path, &mut retired);
        ShaderCacheOutput { entry, retired }
    }
}
