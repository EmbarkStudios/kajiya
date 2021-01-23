#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked
#include "../inc/gbuffer.hlsl"
#include "../inc/color.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] RWByteAddressBuffer surfel_meta_buf;
[[vk::binding(3)]] RWByteAddressBuffer surfel_hash_key_buf;
[[vk::binding(4)]] RWByteAddressBuffer surfel_hash_value_buf;
[[vk::binding(5)]] ByteAddressBuffer cell_index_offset_buf;
[[vk::binding(6)]] ByteAddressBuffer surfel_index_buf;
[[vk::binding(7)]] StructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(8)]] StructuredBuffer<float4> surfel_irradiance_buf;
[[vk::binding(9)]] RWTexture2D<uint2> tile_surfel_alloc_tex;
[[vk::binding(10)]] RWTexture2D<float4> debug_out_tex;

[[vk::binding(11)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

#include "surfel_grid_hash_mut.hlsl"

groupshared uint gs_px_score_loc_packed;

float inverse_lerp(float minv, float maxv, float v) {
    return (v - minv) / (maxv - minv);
}


[numthreads(8, 8, 1)]
void main(
    uint2 px: SV_DispatchThreadID,
    uint idx_within_group: SV_GroupIndex,
    uint2 group_id: SV_GroupID,
    uint2 px_within_group: SV_GroupThreadID
) {
    if (0 == idx_within_group) {
        gs_px_score_loc_packed = 0;
        tile_surfel_alloc_tex[group_id] = uint2(0, 0);
    }

    GroupMemoryBarrierWithGroupSync();

    uint seed = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);

    //const float4 output_tex_size = float4(1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0);
    const float2 uv = get_uv(px, gbuffer_tex_size);

    debug_out_tex[px] = 0.0.xxxx;

    const float4 gbuffer_packed = gbuffer_tex[px];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        return;
    }

    const float z_over_w = depth_tex[px];
    const float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    const float4 pt_vs = mul(frame_constants.view_constants.sample_to_view, pt_cs);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, pt_vs);
    pt_ws /= pt_ws.w;

    const float pt_depth = -pt_vs.z / pt_vs.w;

    // TODO: nuke
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();

    SurfelGridHashEntry entry = surfel_hash_lookup_by_grid_coord(surfel_pos_to_grid_coord(pt_ws.xyz));

    const float2 group_center_offset = float2(px_within_group) - 3.5;

    float px_score = 1e10 / (1.0 + dot(group_center_offset, group_center_offset));

    uint cell_idx = 0xffffffff;

    if (entry.found) {
        px_score = 0.0;

        cell_idx = surfel_hash_value_buf.Load(sizeof(uint) * entry.idx);
        float3 surfel_color = 0;//uint_id_to_color(cell_idx) * 0.3;

        // Calculate px score based on surrounding surfels

        uint2 surfel_idx_loc_range = cell_index_offset_buf.Load2(sizeof(uint) * cell_idx);
        const uint cell_surfel_count = surfel_idx_loc_range.y - surfel_idx_loc_range.x;

        // TEMP HACK: Make sure we're not iterating over tons of surfels out of bounds
        surfel_idx_loc_range.y = min(surfel_idx_loc_range.y, surfel_idx_loc_range.x + 128);

        float3 total_color = 0.0.xxx;
        float total_weight = 0.0;
        float scoring_total_weight = 0.0;
        uint useful_surfel_count = 0;

        for (uint surfel_idx_loc = surfel_idx_loc_range.x; surfel_idx_loc < surfel_idx_loc_range.y; ++surfel_idx_loc) {
            const uint surfel_idx = surfel_index_buf.Load(sizeof(uint) * surfel_idx_loc);

            Vertex surfel = unpack_vertex(surfel_spatial_buf[surfel_idx]);
            //surfel_color = (surfel.normal * 0.5 + 0.5) * 0.3;

            float4 surfel_irradiance_packed = surfel_irradiance_buf[surfel_idx];
            surfel_color = surfel_irradiance_packed.xyz / max(1.0, surfel_irradiance_packed.w);

            const float3 pos_offset = pt_ws.xyz - surfel.position.xyz;
            const float directional_weight = pow(max(0.0, dot(surfel.normal, gbuffer.normal)), 2);
            const float dist = length(pos_offset);
            const float mahalanobis_dist = length(pos_offset) * (1 + abs(dot(pos_offset, surfel.normal)) * SURFEL_NORMAL_DIRECTION_SQUISH);

            static const float RADIUS_OVERSCALE = 1.25;

            float weight = smoothstep(
                SURFEL_RADIUS * RADIUS_OVERSCALE,
                0.0,
                mahalanobis_dist) * directional_weight;

            const float scoring_weight = smoothstep(
                SURFEL_RADIUS,
                0.0,
                mahalanobis_dist) * directional_weight;

            useful_surfel_count += scoring_weight > 1e-5 ? 1 : 0;

            weight *= saturate(inverse_lerp(31.0, 128.0, surfel_irradiance_packed.w));
            total_weight += weight;
            scoring_total_weight += scoring_weight;
            total_color += surfel_color * weight;// * (dist < 0.05 ? 10 : 0);
        }

        total_color /= max(0.1, total_weight);

        #if 0
            total_color =
                cell_surfel_count > 32 ? float3(1, 0, 0):
                cell_surfel_count > 16 ? float3(1, 1, 0):
                cell_surfel_count > 8 ? float3(0, 1, 0):
                float3(0, 1, 0);
        #endif

        #if 0
            total_color = lerp(float3(1, 0, 0), float3(0, 1, 0), useful_surfel_count / 10.0);
        #endif

        if (cell_surfel_count > 128) {
            total_color = float3(1, 0, 1);
        }

        //total_color = uint_id_to_color(cell_idx) * 0.3;
        //total_color = saturate(1.0 - length(pt_ws.xyz));

        debug_out_tex[px] = float4(total_color, 1);

        if (cell_surfel_count >= 32 || useful_surfel_count > 2 || scoring_total_weight > 0.2) {
            return;
        }

        px_score = 1.0 / (1.0 + total_weight);
    } else {
        if (entry.vacant) {
            //if (uint_to_u01_float(hash1_mut(seed)) < 0.001) {
                if (entry.acquire()) {
                    surfel_meta_buf.InterlockedAdd(0 * sizeof(uint), 1, cell_idx);
                    surfel_hash_value_buf.Store(entry.idx * 4, cell_idx);
                } else {
                    // Allocating the cell
                    //debug_out_tex[px] = float4(10, 0, 0, 1);
                    return;
                }
            //}
        } else {
            // Too many conflicts; cannot insert a new entry.
            debug_out_tex[px] = float4(10, 0, 10, 1);
            return;
        }
    }

    // Execution only survives here if we would like to allocate a surfel in this tile

    uint px_score_loc_packed = 0;
    if (uint_to_u01_float(hash1_mut(seed)) < 0.001 * pt_depth / 64.0) {
        px_score_loc_packed = (asuint(px_score) & (0xffffffff - 63)) | (px_within_group.y * 8 + px_within_group.x);
    }
    
    InterlockedMax(gs_px_score_loc_packed, px_score_loc_packed);
    GroupMemoryBarrierWithGroupSync();

    uint group_id_hash = hash2(group_id);
    //out_color = uint_id_to_color(group_id_hash) * 0.1;

    if (gs_px_score_loc_packed == px_score_loc_packed && px_score_loc_packed != 0) {
        debug_out_tex[px] = float4(10, 0, 0, 1);
        tile_surfel_alloc_tex[group_id] = uint2(px_score_loc_packed, cell_idx);
    } else {
        //debug_out_tex[px] = float4(0, 0, 10, 1);
    }
}
