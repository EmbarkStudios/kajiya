// Must match `wrc.rs`
static const int3 WRC_GRID_DIMS = int3(4, 2, 5);
static const int WRC_PROBE_DIMS = 32;
static const int2 WRC_ATLAS_PROBE_COUNT = int2(16, 16);
static const float3 WRC_GRID_CENTER = float3(WRC_GRID_DIMS - 1) * 0.5;

float3 wrc_probe_center(int3 probe_idx) {
    return probe_idx - WRC_GRID_CENTER;
}

uint2 wrc_probe_idx_to_atlas_tile(uint probe_idx) {
    return uint2(probe_idx % WRC_ATLAS_PROBE_COUNT.x, probe_idx / WRC_ATLAS_PROBE_COUNT.x);
}

uint3 wrc_probe_idx_to_coord(uint probe_idx) {
    return uint3(
        probe_idx % WRC_GRID_DIMS.x * WRC_GRID_DIMS.y,
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

/*

    const uint2 atlas_px = tile * WRC_PROBE_DIMS + probe_px;

    const uint3 probe_coord = uint3(
        probe_idx % WRC_GRID_DIMS.x * WRC_GRID_DIMS.y,
        (probe_idx / WRC_GRID_DIMS.x) % WRC_GRID_DIMS.y,
        probe_idx / (WRC_GRID_DIMS.x * WRC_GRID_DIMS.y)
    );
*/