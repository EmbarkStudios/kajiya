## Using DLSS

DLSS is supported on Nvidia RTX GPUs, and `kajiya` can currently use it when running on Windows.

#### SDK

Nvidia's DLSS EULA prohibits distribution of the DLSS SDK, so you will have to obtain it yourself. The stand-alone SDK currently requires an NVIDIA Developer Program membership, _however_ the Unreal Enigine 5 plugin does not, yet it contains the necessary files.

Therefore, the easiest way to get DLSS into `kajiya` is to [download the UE5 DLSS plugin](https://developer.nvidia.com/dlss-getting-started#ue-version), and extract the following:

* Copy `DLSS/Binaries/ThirdParty/Win64/nvngx_dlss.dll` to the root `kajiya` folder (where this README resides).
* Copy the entire `DLSS/Source/ThirdParty/NGX` folder to `crates/lib/ngx_dlss/NGX`

#### Rust bindings

Please make sure you can run `bindgen`, which is necessary to generate a Rust binding to the SDK. Here's the official [installation instructions and requirements page](https://rust-lang.github.io/rust-bindgen/requirements.html). If `cargo` complains about `libclang.dll`, it's probably this.

#### Usage

When building `kajiya`, use the `dlss` Cargo feature, and specify temporal upsampling, e.g.:

```
cargo run --bin view --release --features dlss -- --scene battle --no-debug --temporal-upsampling 1.5 --width 1920 --height 1080
```

This will run DLSS _Quality_ mode. `--temporal-upsampling 2.0` corresponds to _Performance_.

Please note that while DLSS works well for AAA-style content, it currently struggles with flat colors and smooth gradients. The built-in `kajiya` TAA and its temporal upsampling tends to look better there.
