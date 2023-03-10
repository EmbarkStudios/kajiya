use crate::file::LoadFile;
use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use relative_path::RelativePathBuf;
use std::{path::PathBuf, sync::Arc};
use turbosloth::*;

pub struct CompiledShader {
    pub name: String,
    pub spirv: Bytes,
}

#[derive(Clone, Hash)]
pub struct CompileShader {
    pub path: PathBuf,
    pub profile: String,
}

#[async_trait]
impl LazyWorker for CompileShader {
    type Output = Result<CompiledShader>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let ext = self
            .path
            .extension()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "".to_string());

        let name = self
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        match ext.as_str() {
            "glsl" => unimplemented!(),
            "spv" => {
                let spirv = LoadFile::new(self.path.clone())?.run(ctx).await?;
                Ok(CompiledShader { name, spirv })
            }
            "hlsl" => {
                let file_path = self.path.to_str().unwrap().to_owned();
                let source = shader_prepper::process_file(
                    &file_path,
                    &mut ShaderIncludeProvider { ctx },
                    String::new(),
                );
                let source = source
                    .map_err(|err| anyhow!("{}", err))
                    .with_context(|| format!("shader path: {:?}", self.path))?;
                let target_profile = format!("{}_6_4", self.profile);
                let spirv = compile_generic_shader_hlsl_impl(&name, &source, &target_profile)?;

                Ok(CompiledShader { name, spirv })
            }
            _ => anyhow::bail!("Unrecognized shader file extension: {}", ext),
        }
    }
}

pub struct RayTracingShader {
    pub name: String,
    pub spirv: Bytes,
}

#[derive(Clone, Hash)]
pub struct CompileRayTracingShader {
    pub path: PathBuf,
}

#[async_trait]
impl LazyWorker for CompileRayTracingShader {
    type Output = Result<RayTracingShader>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let file_path = self.path.to_str().unwrap().to_owned();
        let source = shader_prepper::process_file(
            &file_path,
            &mut ShaderIncludeProvider { ctx },
            String::new(),
        );
        let source = source.map_err(|err| anyhow!("{}", err))?;

        let ext = self
            .path
            .extension()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "".to_string());

        let name = self
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        match ext.as_str() {
            "glsl" => unimplemented!(),
            "hlsl" => {
                let target_profile = "lib_6_4";
                let spirv = compile_generic_shader_hlsl_impl(&name, &source, target_profile)?;

                Ok(RayTracingShader { name, spirv })
            }
            _ => anyhow::bail!("Unrecognized shader file extension: {}", ext),
        }
    }
}

struct ShaderIncludeProvider {
    ctx: RunContext,
}

impl shader_prepper::IncludeProvider for ShaderIncludeProvider {
    type IncludeContext = String;

    fn get_include(
        &mut self,
        path: &str,
        parent_file: &Self::IncludeContext,
    ) -> std::result::Result<
        (String, Self::IncludeContext),
        shader_prepper::BoxedIncludeProviderError,
    > {
        let resolved_path = if let Some('/') = path.chars().next() {
            path.to_owned()
        } else {
            let mut folder: RelativePathBuf = parent_file.into();
            folder.pop();
            folder.join(path).as_str().to_string()
        };

        let blob: Arc<Bytes> = smol::block_on(
            crate::file::LoadFile::new(&resolved_path)
                .with_context(|| format!("Failed loading shader include {}", path))?
                .into_lazy()
                .eval(&self.ctx),
        )?;

        Ok((String::from_utf8(blob.to_vec())?, resolved_path))
    }
}

pub fn get_cs_local_size_from_spirv(spirv: &[u32]) -> Result<[u32; 3]> {
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(spirv, &mut loader).unwrap();
    let module = loader.module();

    for inst in module.global_inst_iter() {
        //if spirv_headers::Op::ExecutionMode == inst.class.opcode {
        if inst.class.opcode as u32 == 16 {
            let local_size = &inst.operands[2..5];
            use rspirv::dr::Operand::LiteralInt32;

            if let [LiteralInt32(x), LiteralInt32(y), LiteralInt32(z)] = *local_size {
                return Ok([x, y, z]);
            } else {
                bail!("Could not parse the ExecutionMode SPIR-V op");
            }
        }
    }

    Err(anyhow!("Could not find a ExecutionMode SPIR-V op"))
}

fn compile_generic_shader_hlsl_impl(
    name: &str,
    source: &[shader_prepper::SourceChunk],
    target_profile: &str,
) -> Result<Bytes> {
    let mut source_text = String::new();
    for s in source {
        source_text += &s.source;
    }

    let t0 = std::time::Instant::now();
    let spirv = hassle_rs::compile_hlsl(
        name,
        &source_text,
        "main",
        target_profile,
        &[
            "-spirv",
            //"-enable-16bit-types",
            "-fspv-target-env=vulkan1.2",
            "-WX",      // warnings as errors
            "-Ges",     // strict mode
            "-HV 2021", // HLSL version 2021
        ],
        &[],
    )
    .map_err(|err| anyhow!("{}", err))?;

    log::trace!("dxc took {:?} for {}", t0.elapsed(), name,);

    Ok(spirv.into())
}
