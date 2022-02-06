#ifndef SURFEL_GI_BINDINGS_HLSL
#define SURFEL_GI_BINDINGS_HLSL

#include "../inc/mesh.hlsl" // for VertexPacked

#define DEFINE_SURFEL_GI_BINDINGS(b0, b1, b2, b3, b4, b5, b6) \
    [[vk::binding(b0)]] ByteAddressBuffer surfel_hash_key_buf; \
    [[vk::binding(b1)]] ByteAddressBuffer surfel_hash_value_buf; \
    [[vk::binding(b2)]] ByteAddressBuffer cell_index_offset_buf; \
    [[vk::binding(b3)]] ByteAddressBuffer surfel_index_buf; \
    [[vk::binding(b4)]] StructuredBuffer<VertexPacked> surfel_spatial_buf; \
    [[vk::binding(b5)]] StructuredBuffer<float4> surfel_irradiance_buf; \
    [[vk::binding(b6)]] RWStructuredBuffer<uint> surfel_life_buf;

#endif  // SURFEL_GI_BINDINGS_HLSL