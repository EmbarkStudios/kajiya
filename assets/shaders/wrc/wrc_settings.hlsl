#ifndef WRC_SETTINGS_HLSL
#define WRC_SETTINGS_HLSL

// Must match `wrc.rs`
static const int3 WRC_GRID_DIMS = int3(8, 3, 8);
static const int WRC_PROBE_DIMS = 32;
static const int2 WRC_ATLAS_PROBE_COUNT = int2(16, 16);
static const float3 WRC_GRID_WORLD_SIZE = float3(WRC_GRID_DIMS);
static const float WRC_MIN_TRACE_DIST = M_CBRT_2;

float3 wrc_grid_center() {
    //return float3(0, 0.4 + sin(frame_constants.frame_index * 0.015) * 0.5, 0);
    //return float3(0, 1.5, 0);
    //return float3(0, -0.5, 0);
    //return float3(0, 1, -15);
    //return float3(0, 0, 14);
    //return float3(0, 0.0, 8);
    //return float3(0, -100.0, 0);
    return float3(-3.3786697, 0, -37.03225);
}

float3 wrc_probe_center(int3 probe_idx) {
    return probe_idx - WRC_GRID_DIMS * 0.5 + 0.5 + wrc_grid_center();
}

uint2 wrc_probe_idx_to_atlas_tile(uint probe_idx) {
    return uint2(probe_idx % WRC_ATLAS_PROBE_COUNT.x, probe_idx / WRC_ATLAS_PROBE_COUNT.x);
}

uint3 wrc_probe_idx_to_coord(uint probe_idx) {
    return uint3(
        probe_idx % WRC_GRID_DIMS.x,
        (probe_idx / WRC_GRID_DIMS.x) % WRC_GRID_DIMS.y,
        probe_idx / (WRC_GRID_DIMS.x * WRC_GRID_DIMS.y)
    );
}

uint probe_coord_to_idx(uint3 probe_coord) {
    return
        probe_coord.x
        + probe_coord.y * WRC_GRID_DIMS.x
        + probe_coord.z * (WRC_GRID_DIMS.x * WRC_GRID_DIMS.y);
}

int3 wrc_world_pos_to_coord(float3 pos) {
    // TODO: scaling
    return int3(floor(pos - wrc_grid_center() + WRC_GRID_DIMS * 0.5));
}

float3 wrc_world_pos_to_interp_frac(float3 pos) {
    // TODO: scaling
    return frac(pos - wrc_grid_center() + WRC_GRID_DIMS * 0.5);
}

/*

    const uint2 atlas_px = tile * WRC_PROBE_DIMS + probe_px;

    const uint3 probe_coord = uint3(
        probe_idx % WRC_GRID_DIMS.x * WRC_GRID_DIMS.y,
        (probe_idx / WRC_GRID_DIMS.x) % WRC_GRID_DIMS.y,
        probe_idx / (WRC_GRID_DIMS.x * WRC_GRID_DIMS.y)
    );
*/

#endif  // WRC_SETTINGS_HLSL