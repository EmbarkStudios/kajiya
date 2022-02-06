#ifndef SURFEL_GI_LOOKUP_HLSL
#define SURFEL_GI_LOOKUP_HLSL

#include "surfel_grid_hash.hlsl"

float3 lookup_surfel_gi(float3 pt_ws, float3 normal_ws) {
    const SurfelGridHashEntry entry = surfel_hash_lookup_by_grid_coord(surfel_pos_to_grid_coord(pt_ws));
    if (!entry.found) {
        return 0.0.xxx;
    }

    const uint cell_idx = surfel_hash_value_buf.Load(sizeof(uint) * entry.idx);
    uint2 surfel_idx_loc_range = cell_index_offset_buf.Load2(sizeof(uint) * cell_idx);
    const uint cell_surfel_count = surfel_idx_loc_range.y - surfel_idx_loc_range.x;

    // TEMP HACK: Make sure we're not iterating over tons of surfels out of bounds
    surfel_idx_loc_range.y = min(surfel_idx_loc_range.y, surfel_idx_loc_range.x + 128);

    float3 total_color = 0.0.xxx;
    float total_weight = 0.0;

    for (uint surfel_idx_loc = surfel_idx_loc_range.x; surfel_idx_loc < surfel_idx_loc_range.y; ++surfel_idx_loc) {
        const uint surfel_idx = surfel_index_buf.Load(sizeof(uint) * surfel_idx_loc);
        Vertex surfel = unpack_vertex(surfel_spatial_buf[surfel_idx]);

        const float4 surfel_irradiance_packed = surfel_irradiance_buf[surfel_idx];
        float3 surfel_color = surfel_irradiance_packed.xyz;

        const float3 pos_offset = pt_ws.xyz - surfel.position.xyz;
        const float directional_weight = max(0.0, dot(surfel.normal, normal_ws));
        const float dist = length(pos_offset);
        const float mahalanobis_dist = length(pos_offset) * (1 + abs(dot(pos_offset, surfel.normal)) * SURFEL_NORMAL_DIRECTION_SQUISH);

        static const float RADIUS_OVERSCALE = 1.25;

        const float surfel_radius = surfel_radius_for_pos(surfel.position);
        float weight = smoothstep(
            surfel_radius * RADIUS_OVERSCALE,
            0.0,
            mahalanobis_dist) * directional_weight;

        total_weight += weight;
        total_color += surfel_color * weight;
    }

    total_color /= max(0.1, total_weight);
    return total_color;
}

#endif // SURFEL_GI_LOOKUP_HLSL
