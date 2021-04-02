use anyhow::Context as _;
use hotwatch::Hotwatch;
use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::{fs::File, path::PathBuf};
use turbosloth::*;

lazy_static! {
    static ref FILE_WATCHER: Mutex<Hotwatch> =
        Mutex::new(Hotwatch::new_with_custom_delay(std::time::Duration::from_millis(100)).unwrap());
}

lazy_static! {
    static ref VFS_MOUNT_POINT: Mutex<Option<PathBuf>> = Mutex::new(None);
}

pub fn set_vfs_mount_point(path: impl Into<PathBuf>) {
    *VFS_MOUNT_POINT.lock() = Some(path.into());
}

pub fn canonical_path_from_vfs(path: impl Into<PathBuf>) -> std::io::Result<PathBuf> {
    let mut path = path.into();

    if let Ok(rel_path) = path.strip_prefix("/") {
        if let Some(vfs_mount_point) = VFS_MOUNT_POINT.lock().as_ref() {
            path = vfs_mount_point.join(rel_path);
        } else {
            path = rel_path.to_owned();
        }
    }

    path.canonicalize()
}

#[derive(Clone, Hash)]
pub struct LoadFile {
    path: PathBuf,
}

impl LoadFile {
    pub fn new(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = canonical_path_from_vfs(path)?;
        Ok(Self { path })
    }
}

#[async_trait]
impl LazyWorker for LoadFile {
    type Output = anyhow::Result<Vec<u8>>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let invalidation_trigger = ctx.get_invalidation_trigger();

        FILE_WATCHER
            .lock()
            .watch(self.path.clone(), move |event| {
                if matches!(event, hotwatch::Event::Write(_)) {
                    invalidation_trigger();
                }
            })
            .with_context(|| format!("LazyWorker: trying to watch {:?}", self.path))?;

        let mut buffer = Vec::new();
        std::io::Read::read_to_end(&mut File::open(&self.path)?, &mut buffer)
            .with_context(|| format!("LazyWorker: trying to read {:?}", self.path))?;

        Ok(buffer)
    }

    fn debug_description(&self) -> Option<std::borrow::Cow<'static, str>> {
        Some(format!("LoadFile({:?})", self.path).into())
    }
}
