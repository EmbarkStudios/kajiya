#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked
#include "../inc/color.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] RWByteAddressBuffer surfel_meta_buf;
[[vk::binding(3)]] RWByteAddressBuffer surfel_hash_key_buf;
[[vk::binding(4)]] RWByteAddressBuffer surfel_hash_value_buf;
[[vk::binding(5)]] RWByteAddressBuffer cell_index_offset_buf;
[[vk::binding(6)]] RWByteAddressBuffer surfel_index_buf;
[[vk::binding(7)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(8)]] RWTexture2D<uint2> tile_surfel_alloc_tex;
[[vk::binding(9)]] RWTexture2D<float4> debug_out_tex;

#include "surfel_grid_hash.hlsl"

groupshared uint gs_px_score_loc_packed;


[numthreads(8, 8, 1)]
void main(
    uint2 px: SV_DispatchThreadID,
    uint thread_index: SV_GroupIndex,
    uint2 group_id: SV_GroupID,
    uint2 px_within_group: SV_GroupThreadID
) {
    if (0 == thread_index) {
        gs_px_score_loc_packed = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    uint seed = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);

    float4 output_tex_size = float4(1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0);
    float2 uv = get_uv(px, output_tex_size);

    float4 gbuffer_packed = gbuffer_tex[px];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        debug_out_tex[px] = 0.0.xxxx;
        return;
    }

    float z_over_w = depth_tex[px];
    float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, mul(frame_constants.view_constants.sample_to_view, pt_cs));
    pt_ws /= pt_ws.w;

    SurfelGridHashEntry entry = surfel_hash_lookup(pt_ws.xyz);

    const float2 group_center_offset = float2(px_within_group) - 3.5;

    float px_score = 1e10 / (1.0 + dot(group_center_offset, group_center_offset));
    float3 out_color = 0.5.xxx;

    if (entry.found) {
        const uint cell_idx = surfel_hash_value_buf.Load(entry.idx * 4);
        out_color = uint_id_to_color(cell_idx) * float(cell_idx) / 100000.0;

        // TODO: calculate px score based on surrounding surfels
    } else {
        if (entry.vacant) {
            //if (uint_to_u01_float(hash1_mut(seed)) < 0.001) {
                if (entry.acquire()) {
                    uint cell_idx = 0;
                    surfel_meta_buf.InterlockedAdd(0, 1, cell_idx);

                    surfel_hash_value_buf.Store(entry.idx * 4, cell_idx);
                }
            //}
        } else {
            // Too many conflicts; cannot insert a new entry.
            debug_out_tex[px] = float4(10, 0, 10, 1);
            return;
        }
    }

    const uint px_score_loc_packed = (asuint(px_score) & (0xffffffff - 63)) | (px_within_group.y * 8 + px_within_group.x);

    InterlockedMax(gs_px_score_loc_packed, px_score_loc_packed);
    GroupMemoryBarrierWithGroupSync();

    uint group_id_hash = hash2(group_id);
    //out_color = uint_id_to_color(group_id_hash) * 0.1;

    if (gs_px_score_loc_packed == px_score_loc_packed) {
        out_color = float3(10, 0, 0);
    }

    debug_out_tex[px] = float4(out_color, 1);
}
