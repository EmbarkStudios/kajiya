use std::{collections::HashMap, fs::File};

use parking_lot::Mutex;

lazy_static::lazy_static! {
    static ref ASSET_MMAPS: Mutex<HashMap<String, memmap2::Mmap>> = Mutex::new(HashMap::new());
}

pub fn mmapped_asset<T>(path: &str) -> anyhow::Result<&'static T> {
    let mut mmaps = ASSET_MMAPS.lock();
    let data: &[u8] = mmaps.entry(path.to_owned()).or_insert_with(|| {
        let file = File::open(path).unwrap();
        unsafe { memmap2::MmapOptions::new().map(&file).unwrap() }
    });
    let asset: &T = unsafe { (data.as_ptr() as *const T).as_ref() }.unwrap();
    Ok(asset)
}
