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
[[vk::binding(7)]] RWStructuredBuffer<VertexPacked> surfel_reposition_proposal_buf;
[[vk::binding(8)]] RWStructuredBuffer<float4> surfel_aux_buf;
[[vk::binding(9)]] RWStructuredBuffer<float4> surfel_irradiance_buf;
[[vk::binding(10)]] RWStructuredBuffer<uint> surfel_life_buf;
[[vk::binding(11)]] RWStructuredBuffer<uint> surfel_pool_buf;
[[vk::binding(12)]] Texture2D<uint2> tile_surfel_alloc_tex;
[[vk::binding(13)]] Texture2D<float4> tile_surfel_irradiance_tex;
[[vk::binding(14)]] cbuffer _ {
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

    if (!all(pt_ws == pt_ws)) {
        // Batman, Batman, Batman...
        return;
    }

    float4 gbuffer_packed = gbuffer_tex[px];

    const float3 eye_pos = get_eye_position();
    const uint4 c4_coord = surfel_grid_coord_to_c4(surfel_pos_to_grid_coord(pt_ws.xyz, eye_pos));
    const float3 c4_center = surfel_grid_coord_center(c4_coord, eye_pos);
    
    VertexPacked surfel;
    // TODO: proper packing
    //surfel.data0 = float4(pt_ws.xyz, gbuffer_packed.y);
    surfel.data0 = float4(c4_center, gbuffer_packed.y);

    uint surfel_alloc_idx;
    surfel_meta_buf.InterlockedAdd(SURFEL_META_ALLOC_COUNT, 1, surfel_alloc_idx);

    const uint surfel_idx = surfel_pool_buf[surfel_alloc_idx];
    surfel_meta_buf.InterlockedMax(SURFEL_META_SURFEL_COUNT, surfel_idx + 1);

    // Clear dead state, mark used.
    surfel_life_buf[surfel_idx] = 0;

    surfel_spatial_buf[surfel_idx] = surfel;
    surfel_reposition_proposal_buf[surfel_idx] = surfel;

    // Irradiance at the interpolated surfel position
    const float4 source_irradiance = tile_surfel_irradiance_tex[tile_px];

    // Starting radiance and sample count for the new surfel
    const float4 start_irradiance_and_sample_count = max(0.0, float4(
        source_irradiance.rgb,
        //float3(1, 0, 0),
        min(64, 32 * source_irradiance.a)
    ));

    surfel_aux_buf[surfel_idx * 2 + 0] = start_irradiance_and_sample_count;
    surfel_irradiance_buf[surfel_idx] = start_irradiance_and_sample_count;
}
