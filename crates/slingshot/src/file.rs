use anyhow::Context as _;
use hotwatch::Hotwatch;
use lazy_static::lazy_static;
use std::{fs::File, path::PathBuf, sync::Mutex};
use turbosloth::*;

lazy_static! {
    static ref FILE_WATCHER: Mutex<Hotwatch> =
        Mutex::new(Hotwatch::new_with_custom_delay(std::time::Duration::from_millis(200)).unwrap());
}

#[derive(Clone, Hash)]
pub struct LoadFile {
    pub path: PathBuf,
}

#[async_trait]
impl LazyWorker for LoadFile {
    type Output = anyhow::Result<Vec<u8>>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let invalidation_trigger = ctx.get_invalidation_trigger();

        FILE_WATCHER
            .lock()
            .unwrap()
            .watch(self.path.clone(), move |_| {
                invalidation_trigger();
            })
            .with_context(|| format!("LazyWorker: trying to watch {:?}", self.path))?;

        let mut buffer = Vec::new();
        std::io::Read::read_to_end(&mut File::open(&self.path)?, &mut buffer)
            .with_context(|| format!("LazyWorker: trying to read {:?}", self.path))?;

        Ok(buffer)
    }
}
