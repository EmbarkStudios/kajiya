use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    // Find the ngx lib in this crate rather than in the including project
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    println!(
        "cargo:rustc-link-search=native={}",
        Path::new(&dir).join("NGX/Lib/x64").display()
    );

    println!("cargo:rustc-link-lib=nvsdk_ngx_d");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    let vulkan_sdk = env::var("VULKAN_SDK").unwrap_or_else(|_| {
        panic!("The environment variable `VULKAN_SDK` was not found. Is the Vulkan SDK installed?")
    });

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}/Include/Vulkan", vulkan_sdk))
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .allowlist_function("NVSDK.*")
        .allowlist_type("NVSDK.*")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
