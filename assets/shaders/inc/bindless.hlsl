#include "mesh.hlsl"

[[vk::binding(0, 1)]] StructuredBuffer<Mesh> meshes;
[[vk::binding(1, 1)]] ByteAddressBuffer vertices;
[[vk::binding(2, 1)]] Texture2D bindless_textures[];
