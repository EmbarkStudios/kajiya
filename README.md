<!-- Allow this file to not have a first line heading -->
<!-- markdownlint-disable-file MD041 -->

<!-- inline html -->
<!-- markdownlint-disable-file MD033 -->

<div align="center">
   
# 💡 kajiya

**Experimental real-time global illumination renderer made with Rust and Vulkan**

[![Embark](https://img.shields.io/badge/embark-open%20source-blueviolet.svg)](https://embark.dev)
[![Embark](https://img.shields.io/badge/discord-ark-%237289da.svg?logo=discord)](https://discord.gg/dAuKfZS)
[![dependency status](https://deps.rs/repo/github/EmbarkStudios/kajiya/status.svg)](https://deps.rs/repo/github/EmbarkStudios/kajiya)
[![Build status](https://github.com/EmbarkStudios/kajiya/workflows/CI/badge.svg)](https://github.com/EmbarkStudios/kajiya/actions)
</div>

Its general goal is to get as close as possible to path-traced reference at real-time rates in dynamic scenes, without any precomputed light transport, or manually placed light probes.

`kajiya` does not currently aim to be a fully-featured renderer used to ship games, support all sorts of scenes, lighting phenomena, or a wide range of hardware. It's a hobby project, takes a lot of shortcuts, and is perpetually a work in progress.

For more context, check out our [announcement article on Embark's Medium](https://medium.com/embarkstudios/homegrown-rendering-with-rust-1e39068e56a7). You'll also get to learn how `kajiya` connects to our rendering work, and the [`rust-gpu`](https://github.com/EmbarkStudios/rust-gpu) project!

![image (5)](https://user-images.githubusercontent.com/16522064/146789417-0cc84f60-157d-4a7d-99f5-79122c1fa982.png)
_Ruins environment rendered in kajiya. [Scene](https://www.unrealengine.com/marketplace/en-US/product/modular-ruins-c) by Crebotoly_

## Features

* Hybrid rendering using a mixture of raster, compute, and ray-tracing
* Dynamic global illumination
    * Multi-bounce temporally-recurrent voxel-based diffuse
    * Short-range ray-traced diffuse for high-frequency details
    * Single bounce specular, falling back to diffuse after the first hit
* Sun with ray-traced soft shadows
* Standard PBR with GGX and roughness/metalness
    * Energy-preserving multi-scattering BRDF
* Reference path-tracing mode
* Temporal super-resolution and anti-aliasing
* Natural tone mapping
* Physically-based glare
* Basic motion blur
* Contrast-adaptive sharpening
* Optional DLSS support
* GLTF mesh loading (no animations yet)
* A render graph running it all

## Technical overview

* [A quick presentation](https://docs.google.com/presentation/d/1LWo5TtWUAH9d62sGY9Sjmu1JqIs8BsxLbVDxLuhhX8U/edit?usp=sharing) about the renderer
* Repository highlights:
  * HLSL shaders: [`assets/shaders/`](assets/shaders)
  * Rust shaders: [`crates/lib/rust-shaders/`](crates/lib/rust-shaders)
  * Main render graph passes: [`world_render_passes.rs`](crates/lib/kajiya/src/world_render_passes.rs)
* Notable branches:
  * `restir-meets-surfel` - latest experimental branch, with [new GI in the works](https://gist.github.com/h3r2tic/ba39300c2b2ca4d9ca5f6ff22350a037)

## Platforms

`kajiya` currently works on a limited range of operating systems and hardware.

Hardware:
* Nvidia RTX series
* Nvidia GTX 1060 and newer _with 6+ GB of VRAM_ (slow: driver-emulated ray-tracing)
* AMD Radeon RX 6000 series

Operating systems:
* Windows
* Linux

### (Some) Linux dependencies
* `uuid-dev`
* In case the bundled `libdxcompiler.so` doesn't work: https://github.com/microsoft/DirectXShaderCompiler#downloads

### (Some) MacOS dependencies

* `ossp-uuid` (`brew install ossp-uuid`)

## Building and running

To build `kajiya` and its tools, [you need Rust](https://www.rust-lang.org/tools/install).

There's a very minimal asset pipeline in `bake.rs`, which converts meshes from GLTF to an internal flat format, and calculates texture mips. In order to bake all the provided meshes, run:

* Windows: `bake.cmd`
* Linux: `./bake.sh`

When done, run the renderer demo (`view` app from `crates/bin/view`) via:

* Windows: `build_and_run.cmd [scene_name]`
* Linux: `./build_and_run.sh [scene_name]`

Where `[scene_name]` is one of the file names in `assets/scenes`, without the `.ron` extension, e.g.:

```
build_and_run.cmd battle
```

or

```
cargo run --bin view --release -- --scene battle --width 1920 --height 1080 --no-debug
```

### Controls in the `view` app

* WSAD, QE - movement
* Mouse + RMB - rotate the camera
* Mouse + LMB - rotate the sun
* Shift - move faster
* Ctrl - move slower
* Space - switch to reference path tracing
* Backspace - reset view to previous saved state
* Tab - show/hide the UI

### Resolution scaling

#### DPI

For the `view` app, DPI scaling in the operating system affects the physical number of pixels of the rendering output. The `--width` and `--height` parameters correspond to _logical_ window size **and** the internal rendering resolution. Suppose the OS uses DPI scaling of `1.5`, and the app is launched with `--width 1000`, the actual physical width of the window will be `1500` px. Rendering will still happen at `1000` px, with upscaling to `1500` px at the very end, via a Catmull-Rom kernel.

#### Temporal upsampling

`kajiya` can also render at a reduced internal resolution, and reconstruct a larger image via temporal upsampling, trading quality for performance. A custom temporal super-resolution algorithm is used by default, and [DLSS is supported](docs/using-dlss.md) on some platforms. Both approaches result in better quality than what could be achieved by simply spatially scaling up the image at the end.

For example, `--width 1920 --height 1080 --temporal-upsampling 1.5` will produce a `1920x1080` image by upsampling by a factor of `1.5` from `1280x720`. Most of the rendering will then happen with `1.5 * 1.5 = 2.25` times fewer pixels, resulting in an _almost_ 2x speedup.

## Adding Meshes and Scenes

To add new mesh(es), open `bake.cmd` (Win) / `bake.sh` (Linux), and add

* cargo run --bin bake --release -- --scene "[path]" --scale 1.0 -o [mesh_name]

To add new scenes, in `\assets\scenes`, create a `[scene_name].ron` with the following content:

```
(
    instances: [
        (
            position: (0, 0, 0),
            mesh: "[mesh_name]",
        ),
    ]
)
```

## Technical guides
* [Using DLSS](docs/using-dlss.md)
* [Working on Rust shaders](docs/rust-shaders.md)
* [Using `kajiya` as a crate](docs/using-kajiya.md)

## Known issues

* Vulkan API usage is extremely basic. Resources are usually not released, and barriers aren't optimal.
* There are hard limit on mesh data and instance counts. Exceeding those limits will result in panics and Vulkan validation errors / driver crashes.
* Window (framebuffer) resizing is not yet implemented.
* The voxel GI uses a fixed-size volume around the origin by default.
    * Use `--gi-volume-scale` to change its extent in the `view` app
    * It can be configured to use camera-centered cascades at an extra performance cost (see `CASCADE_COUNT` and `SCROLL_CASCADES` in [`csgi.rs`](../crates/lib/kajiya/src/renderers/csgi.rs`))
* Denoising needs more work (always).

## Acknowledgments

This project is made possible by the awesome open source Rust community, and benefits from a multitude of crates 💖🦀

Special shout-outs go to:

* Felix Westin for his [MinimalAtmosphere](https://github.com/Fewes/MinimalAtmosphere), which this project uses for sky rendering
* AMD, especially Dominik Baumeister and Guillaume Boissé for the [FidelityFX Shadow Denoiser](https://gpuopen.com/fidelityfx-denoiser/), which forms the basis of shadow denoising in `kajiya`.
* Maik Klein for the Vulkan wrapper [ash](https://github.com/MaikKlein/ash), making it easy for `kajiya` to talk to the GPU.
* Traverse Research and Jasper Bekkers for a number of highly relevant crates:
  * Bindings to the DXC shader compiler: [hassle-rs](https://github.com/Traverse-Research/hassle-rs)
  * SPIR-V reflection utilities: [rspirv-reflect](https://github.com/Traverse-Research/rspirv-reflect)
  * Vulkan memory management: [gpu-allocator](https://github.com/Traverse-Research/gpu-allocator)
  * Blue noise sampling: [blue-noise-sampler](https://github.com/Jasper-Bekkers/blue-noise-sampler)

## Contribution

[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v1.4-ff69b4.svg)](../main/CODE_OF_CONDUCT.md)

We welcome community contributions to this project.

Please read our [Contributor Guide](CONTRIBUTING.md) for more information on how to get started.
Please also read our [Contributor Terms](CONTRIBUTING.md#contributor-terms) before you make any contributions.

Any contribution intentionally submitted for inclusion in an Embark Studios project, shall comply with the Rust standard licensing model (MIT OR Apache 2.0) and therefore be dual licensed as described below, without any additional terms or conditions:

### License

This contribution is dual licensed under EITHER OF

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

For clarity, "your" refers to Embark or any other licensee/user of the contribution.
