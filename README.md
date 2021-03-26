# 💡 kajiya

A real-time global illumination renderer; primarily a toy written for fun to explore various algorithms. Uses Vulkan with ray-tracing extensions under the hood.

Its general goal is to get as close as possible to path-traced reference at real-time rates in dynamic scenes, without any precomputed light transport, or manually placed light probes.

At the same time, `kajiya` does not aim to be a fully-featured renderer used to ship games, support all sorts of scenes, lighting phenomena, or a wide range of hardware. It's a hobby project, takes a lot of shortcuts, and is perpetually a work in progress.

![screenshot](docs/screenshot.jpg)

## Features

* Hybrid rendering using a mixture of raster, compute, and ray-tracing
* Dynamic global illumination
    * Multi-bounce temporally-recurrent voxel-based diffuse
    * Short-range ray-traced diffuse for high-frequency details
    * Single bounce specular, falling back to diffuse after the first hit
* Sun with ray-traced shadows (not soft yet)
* Standard PBR with GGX and roughness/metalness
    * Multi-scattering BRDF, energy-preserving metalness
* Reference path-tracing mode
* Sky rendering based on Felix Westin's shaders
* Temporal anti-aliasing
* Natural tone mapping
* Physically-based glare
* Contrast-adaptive sharpening
* GLTF mesh loading (no animations yet)
* A render graph running it all

Not actively used:

* Screen-space ambient occlusion (GTAO)
    * Currently plugged in as a cross-bilateral feature guide for GI denoising
* Screen-space diffuse bounce based on GTAO
    * Still runs, but not displayed by default

Traces of code:

* Basic SDF sculpting and rendering

## Platforms

It currently works on a very limited number of systems and hardware.

Operating systems:
* Windows
* Linux

Hardware:
* Nvidia RTX cards
* Nvidia GTX 1060+ series (slow: driver-emulated ray-tracing)
* Radeon 6000+ _eventually_ - there are probably synchronization issues / bugs on AMD

## Building and running

There's a very minimal asset pipeline in `bake.rs`, which converts meshes from GLTF to an internal flat format, and calculates texture mips. In order to bake all the provided meshes, run:

* Windows: `bake.cmd`
* Linux: `./bake.sh`

When done, run the renderer demo via:

* Windows: `build_and_run.cmd [scene_name]`
* Linux: `./build_and_run.sh [scene_name]`

Where `[scene_name]` is one of the file names in `assets/scenes`, without the `.ron` extension, e.g.:

```
build_and_run.cmd battle
```

or

```
cargo run --bin kajiya --release -- --scene battle --width 1920 --height 1080 --no-debug
```

_Please note that the `smol` async runtime is used for baking and run-time shader compilation. There's no custom executor yet, so the `SMOL_THREADS` environment variable controls parallelism._

### Controls

* WSAD, QE - movement
* Mouse + RMB - rotate the camera
* Mouse + LMB - rotate the sun
* Shift - move faster
* Ctrl - move slower
* Space - switch to reference path tracing
* Backpace - reset reference path tracing accumulation
* Tab - show/hide the UI

## Known issues

* Vulkan API usage is extremely basic. Resources are usually not released, and barriers aren't optimal.
* There's a hard limit on mesh data and instance count. Exceeding those limits will result in Vulkan validation errors / driver crashes.
* Window (framebuffer) resizing is not implemented.
* The voxel GI uses a fixed-size volume around the origin. It will get cascades later.
* Denoising needs more work.
