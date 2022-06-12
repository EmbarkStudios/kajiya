use anyhow::Context as _;
use bytes::Bytes;
use hotwatch::Hotwatch;
use lazy_static::lazy_static;
use normpath::PathExt;
use parking_lot::Mutex;
use std::{collections::HashMap, fs::File, path::PathBuf};
use turbosloth::*;

lazy_static! {
    pub(crate) static ref FILE_WATCHER: Mutex<Hotwatch> =
        Mutex::new(Hotwatch::new_with_custom_delay(std::time::Duration::from_millis(100)).unwrap());
}

lazy_static! {
    static ref VFS_MOUNT_POINTS: Mutex<HashMap<String, PathBuf>> = Mutex::new(
        vec![
            ("/kajiya".to_owned(), PathBuf::from(".")),
            ("/shaders".to_owned(), PathBuf::from("assets/shaders")),
            (
                "/rust-shaders-compiled".to_owned(),
                PathBuf::from("assets/rust-shaders-compiled")
            ),
            ("/images".to_owned(), PathBuf::from("assets/images")),
            ("/cache".to_owned(), PathBuf::from("cache"))
        ]
        .into_iter()
        .collect()
    );
}

pub fn set_vfs_mount_point(mount_point: impl Into<String>, path: impl Into<PathBuf>) {
    VFS_MOUNT_POINTS
        .lock()
        .insert(mount_point.into(), path.into());
}

pub fn set_standard_vfs_mount_points(kajiya_path: impl Into<PathBuf>) {
    let kajiya_path = kajiya_path.into();
    set_vfs_mount_point("/kajiya", &kajiya_path);
    set_vfs_mount_point("/shaders", kajiya_path.join("assets/shaders"));
    set_vfs_mount_point(
        "/rust-shaders-compiled",
        kajiya_path.join("assets/rust-shaders-compiled"),
    );
    set_vfs_mount_point("/images", kajiya_path.join("assets/images"));
}

pub fn canonical_path_from_vfs(path: impl Into<PathBuf>) -> anyhow::Result<PathBuf> {
    let path = path.into();

    for (mount_point, mounted_path) in VFS_MOUNT_POINTS.lock().iter() {
        if let Ok(rel_path) = path.strip_prefix(mount_point) {
            return mounted_path
                .join(rel_path)
                .canonicalize()
                .with_context(|| {
                    format!(
                        "Mounted parent folder: {:?}. Relative path: {:?}",
                        mounted_path, rel_path
                    )
                })
                .with_context(|| format!("canonicalize {:?}", rel_path));
        }
    }

    if path.strip_prefix("/").is_ok() {
        anyhow::bail!(
            "No vfs mount point for {:?}. Current mount points: {:#?}",
            path,
            VFS_MOUNT_POINTS.lock()
        );
    }

    Ok(path)
}

pub fn normalized_path_from_vfs(path: impl Into<PathBuf>) -> anyhow::Result<PathBuf> {
    let path = path.into();

    for (mount_point, mounted_path) in VFS_MOUNT_POINTS.lock().iter() {
        if let Ok(rel_path) = path.strip_prefix(mount_point) {
            return Ok(mounted_path
                .join(rel_path)
                .normalize()
                .with_context(|| {
                    format!(
                        "Mounted parent folder: {:?}. Relative path: {:?}",
                        mounted_path, rel_path
                    )
                })?
                .as_path()
                .to_owned());
        }
    }

    if path.strip_prefix("/").is_ok() {
        anyhow::bail!(
            "No vfs mount point for {:?}. Current mount points: {:#?}",
            path,
            VFS_MOUNT_POINTS.lock()
        );
    }

    Ok(path)
}

#[derive(Clone, Hash)]
pub struct LoadFile {
    path: PathBuf,
}

impl LoadFile {
    pub fn new(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let path = canonical_path_from_vfs(path)?;
        Ok(Self { path })
    }
}

#[async_trait]
impl LazyWorker for LoadFile {
    type Output = anyhow::Result<Bytes>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let invalidation_trigger = ctx.get_invalidation_trigger();

        FILE_WATCHER
            .lock()
            .watch(self.path.clone(), move |event| {
                if matches!(event, hotwatch::Event::Write(_)) {
                    invalidation_trigger();
                }
            })
            .with_context(|| format!("LoadFile: trying to watch {:?}", self.path))?;

        let mut buffer = Vec::new();
        std::io::Read::read_to_end(&mut File::open(&self.path)?, &mut buffer)
            .with_context(|| format!("LoadFile: trying to read {:?}", self.path))?;

        Ok(Bytes::from(buffer))
    }

    fn debug_description(&self) -> Option<std::borrow::Cow<'static, str>> {
        Some(format!("LoadFile({:?})", self.path).into())
    }
}
