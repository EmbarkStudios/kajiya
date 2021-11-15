use anyhow::Context;
use nanoserde::SerJson;
use spirv_builder::{Capability, MetadataPrintout, ModuleResult, SpirvBuilder, SpirvMetadata};

#[derive(SerJson)]
struct RustShaderCompileResult {
    // entry name -> shader path
    entry_to_shader_module: Vec<(String, String)>,
}

fn main() -> anyhow::Result<()> {
    let builder_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let compile_result = SpirvBuilder::new(builder_root.join("../../lib/rust-shaders/"), "spirv-unknown-vulkan1.1")
        .deny_warnings(true)
        .capability(Capability::StorageImageWriteWithoutFormat)
        .capability(Capability::Int8)
        .capability(Capability::RuntimeDescriptorArray)
        .extension("SPV_EXT_descriptor_indexing")
        .print_metadata(MetadataPrintout::None)
        .multimodule(true)
        .spirv_metadata(SpirvMetadata::NameVariables)
        .build()?;

    let target_spv_dir = builder_root.join("../../../assets/rust-shaders-compiled");
    std::fs::create_dir_all(&target_spv_dir).context("Creating the SPIR-V output directory")?;

    // Move all the compiled shaders to the `target_spv_dir`, and create a json file
    // mapping entry points to SPIR-V modules.
    match &compile_result.module {
        ModuleResult::MultiModule(entry_shader) => {
            let res = RustShaderCompileResult {
                entry_to_shader_module: entry_shader
                    .iter()
                    .map(|(entry, src_file)| -> anyhow::Result<(String, String)> {
                        let file_name = src_file.file_name().expect("SPIR-V module file name");
                        let dst_file = target_spv_dir.join(&file_name);

                        // If the compiler detects no changes, it won't generate the output,
                        // so we need to check whether the file actually exists.
                        if src_file.exists() {
                            std::fs::rename(src_file, &dst_file).with_context(|| {
                                format!("Renaming {:?} to {:?}", src_file, dst_file)
                            })?;
                        } else {
                            assert!(dst_file.exists(), "rustc failed to generate SPIR-V module {:?}. Try touching the source files or running `cargo clean` on shaders.", src_file);
                        }

                        Ok((entry.clone(), file_name.to_string_lossy().into()))
                    })
                    .collect::<anyhow::Result<_>>()?,
            };

            std::fs::write(target_spv_dir.join("shaders.json"), res.serialize_json())?;
        }
        _ => panic!(),
    }

    //dbg!(compile_result);

    Ok(())
}
