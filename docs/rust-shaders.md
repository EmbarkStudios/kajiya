# Working on Rust shaders

Add your shaders to the `assets/rust-shaders` crate. Run the shader builder from `assets/rust-shaders/builder` to make sure they can be loaded at runtime. Finally, use `SimpleRenderPass::new_compute_rust` to create a render graph pass (see existing examples).

Rust shaders are compiled asynchronously in the background (unlike HLSL shaders, which use a blocking compilation model). The app will keep running, and once the Rust shaders are ready, they will be reloaded.

Compiled Rust shaders are currently checked into the repository [`assets/rust-shaders-compiled`](../assets/rust-shaders-compiled) as SPIR-V binaries.

To manually build Rust shaders, navigate to [`crates/bin/rust-shader-builder`](../crates/bin/rust-shader-builder) and run `cargo run --release`.
