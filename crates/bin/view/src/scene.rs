#[derive(serde::Deserialize)]
pub struct SceneDesc {
    pub instances: Vec<SceneInstanceDesc>,
}

fn default_instance_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

#[derive(serde::Deserialize)]
pub struct SceneInstanceDesc {
    pub position: [f32; 3],
    #[serde(default = "default_instance_scale")]
    pub scale: [f32; 3],
    #[serde(default)]
    pub rotation: [f32; 3],
    pub mesh: String,
}
