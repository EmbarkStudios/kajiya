#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] RWByteAddressBuffer surfel_meta_buf;
[[vk::binding(3)]] ByteAddressBuffer surfel_hash_key_buf;
[[vk::binding(4)]] ByteAddressBuffer surfel_hash_value_buf;
[[vk::binding(5)]] ByteAddressBuffer surfel_index_buf;
[[vk::binding(6)]] RWStructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(7)]] Texture2D<uint2> tile_surfel_alloc_tex;
[[vk::binding(8)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

#include "surfel_grid_hash.hlsl"

[numthreads(8, 8, 1)]
void main(
    uint2 tile_px: SV_DispatchThreadID,
    uint2 group_id: SV_GroupID,
    uint2 tile_px_within_group: SV_GroupThreadID
) {
    const uint2 tile_surfel_alloc_packed = tile_surfel_alloc_tex[tile_px];
    if (tile_surfel_alloc_packed.x == 0) {
        return;
    }

    const uint px_score_loc_packed = tile_surfel_alloc_packed.x;
    const uint cell_idx = tile_surfel_alloc_packed.y;

    const uint2 px = tile_px * 8 + uint2(px_score_loc_packed & 7, (px_score_loc_packed >> 3) & 7);
    const float2 uv = get_uv(px, gbuffer_tex_size);

    const float z_over_w = depth_tex[px];
    const float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, mul(frame_constants.view_constants.sample_to_view, pt_cs));
    pt_ws /= pt_ws.w;

    float4 gbuffer_packed = gbuffer_tex[px];
    
    VertexPacked surfel;
    // TODO: proper packing
    surfel.data0 = float4(pt_ws.xyz, gbuffer_packed.y);

    uint surfel_idx;
    surfel_meta_buf.InterlockedAdd(1 * sizeof(uint), 1, surfel_idx);

    surfel_spatial_buf[surfel_idx] = surfel;
}