use kajiya_asset::mesh::PackedTriMesh;

use crate::world_renderer::{AddMeshOptions, MeshHandle, WorldRenderer};

impl WorldRenderer {
    pub fn add_baked_mesh(
        &mut self,
        path: impl Into<std::path::PathBuf>,
        opts: AddMeshOptions,
    ) -> anyhow::Result<MeshHandle> {
        Ok(self.add_mesh(
            crate::mmap::mmapped_asset::<PackedTriMesh::Flat, _>(path)?,
            opts,
        ))
    }
}
