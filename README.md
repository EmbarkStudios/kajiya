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

![image](https://user-images.githubusercontent.com/16522064/146711370-2b9b435f-6f19-4b27-8fdd-68964ea9d704.png)
_[The Junk Shop](https://cloud.blender.org/p/gallery/5dd6d7044441651fa3decb56) by Alex Treviño. Original Concept by Anaïs Maamar._

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

Operating systems:
* Windows
* Linux

Hardware:
* Nvidia RTX series
* Nvidia GTX 1060 and newer (slow: driver-emulated ray-tracing)
* AMD Radeon RX 6000 series

## Building and running

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
