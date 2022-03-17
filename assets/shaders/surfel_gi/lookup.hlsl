#ifndef SURFEL_GI_LOOKUP_HLSL
#define SURFEL_GI_LOOKUP_HLSL

#include "surfel_grid_hash.hlsl"

float3 lookup_surfel_gi(float3 pt_ws, float3 normal_ws) {
    const float3 eye_pos = get_eye_position();

    const int3 grid_coord = surfel_pos_to_grid_coord(pt_ws.xyz, eye_pos);
    const uint4 grid_c4 = surfel_grid_coord_to_c4(grid_coord);
    const uint cascade = grid_c4.w;

#if 1
    // Manual trilinear

    const int3 coord_within_cascade = surfel_grid_coord_within_cascade(grid_coord, cascade);
    const float3 center_cell_offset = pt_ws - surfel_grid_coord_center(grid_c4, eye_pos);
    const float cell_diameter = surfel_grid_cell_diameter_in_cascade(cascade);
    const int3 c0 = coord_within_cascade + (center_cell_offset > 0.0.xxx ? (0).xxx : (-1).xxx);
    const float3 interp_t0 = center_cell_offset > 0.0.xxx ? 0.0.xxx : 1.0.xxx;

    float3 irradiance_sum = 0.0.xxx;
    float weight_sum = 0.0;

    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                const int3 sample_within_cascade = c0 + int3(x, y, z);
                const uint4 sample_c4 = uint4(
                    clamp(sample_within_cascade, (int3)0, (int3)(SURFEL_CS - 1)),
                    cascade);
    
                const uint cell_idx = surfel_grid_c4_to_hash(sample_c4);
                const uint4 cell_meta = surf_rcache_grid_meta_buf.Load4(sizeof(uint4) * cell_idx);
                const uint entry_idx = cell_meta.x;

                if (cell_meta.y & SURF_RCACHE_ENTRY_META_OCCUPIED) {
                    float3 interp = center_cell_offset / cell_diameter + interp_t0;
                    interp = saturate((int3(x, y, z) == 0 ? ((1).xxx - interp) : interp));

                    const float3 irradiance = surf_rcache_irradiance_buf[entry_idx].xyz;
                    const float weight = interp.x * interp.y * interp.z;
                    irradiance_sum += irradiance * weight;
                    weight_sum += weight;
                }
            }
        }
    }

    return irradiance_sum / max(1e-10, weight_sum);
#else

    const uint cell_idx = surfel_grid_coord_to_hash(grid_coord);
    const uint4 cell_meta = surf_rcache_grid_meta_buf.Load4(sizeof(uint4) * cell_idx);
    const uint entry_idx = cell_meta.x;

    float4 surfel_irradiance_packed = surf_rcache_irradiance_buf[entry_idx];
    return surfel_irradiance_packed.xyz;
#endif
}

#endif // SURFEL_GI_LOOKUP_HLSL
