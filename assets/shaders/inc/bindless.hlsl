#include "mesh.hlsl"

[[vk::binding(0, 1)]] StructuredBuffer<Mesh> meshes;
[[vk::binding(1, 1)]] ByteAddressBuffer vertices;
#include "bindless_textures.hlsl"
