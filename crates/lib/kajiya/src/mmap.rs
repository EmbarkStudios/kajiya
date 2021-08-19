use std::{collections::HashMap, fs::File, path::PathBuf};

use anyhow::Context;
use parking_lot::Mutex;

lazy_static::lazy_static! {
    static ref ASSET_MMAPS: Mutex<HashMap<PathBuf, memmap2::Mmap>> = Mutex::new(HashMap::new());
}

pub fn mmapped_asset<T, P: Into<std::path::PathBuf>>(path: P) -> anyhow::Result<&'static T> {
    let path = path.into();
    let path = kajiya_backend::canonical_path_from_vfs(&path)
        .with_context(|| format!("Can't mmap asset: file doesn't exist: {:?}", path))?;

    let mut mmaps = ASSET_MMAPS.lock();
    let data: &[u8] = mmaps.entry(path.clone()).or_insert_with(|| {
        let file =
            File::open(&path).unwrap_or_else(|e| panic!("Could not mmap {:?}: {:?}", path, e));
        unsafe { memmap2::MmapOptions::new().map(&file).unwrap() }
    });
    let asset: &T = unsafe { (data.as_ptr() as *const T).as_ref() }.unwrap();
    Ok(asset)
}
