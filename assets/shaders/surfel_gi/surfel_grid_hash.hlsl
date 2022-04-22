#include "../inc/hash.hlsl"
#include "surfel_constants.hlsl"

static const uint MAX_SURFEL_GRID_CELLS = 1024 * 1024 * 2;
static const uint RCACHE_CASCADE_SIZE = 32;
static const uint RCACHE_CASCADE_COUNT = 12;

static const bool SURFEL_GRID_SCROLL = true;
//static const float3 SURFEL_GRID_CENTER = float3(-1.5570648, -1.2360737, 9.283543);
static const float3 SURFEL_GRID_CENTER = float3(0, 0, 14);
//static const float3 SURFEL_GRID_CENTER = float3(-2, 0, -2);
//static const float3 SURFEL_GRID_CENTER = float3(-1.3989427, 0.44028947, -4.080884);
//static const float3 SURFEL_GRID_CENTER = 0.0.xxx;

static const bool RCACHE_USE_NORMAL_BASED_CELL_OFFSET = true;

struct RcacheCoord {
    uint3 coord;
    uint cascade;

    static RcacheCoord from_coord_cascade(uint3 coord, uint cascade) {
        RcacheCoord res;
        res.coord = min(coord, (RCACHE_CASCADE_SIZE - 1).xxx);
        res.cascade = min(cascade, RCACHE_CASCADE_COUNT - 1);
        return res;
    }

    uint cell_idx() {
        return dot(
            uint4(coord, cascade),
            uint4(
                1,
                RCACHE_CASCADE_SIZE,
                RCACHE_CASCADE_SIZE * RCACHE_CASCADE_SIZE,
                RCACHE_CASCADE_SIZE * RCACHE_CASCADE_SIZE * RCACHE_CASCADE_SIZE));    
    }
};

RcacheCoord ws_pos_to_rcache_coord(float3 pos, float3 normal) {
    const float3 center = get_eye_position();

    const uint reserved_cells =
        RCACHE_USE_NORMAL_BASED_CELL_OFFSET
        // Make sure we can actually offset towards the edge of a cascade.
        ? 1
        // Business as usual
        : 0;

    uint cascade; {
        const float3 fcoord = (pos - center) / RCACHE_GRID_CELL_DIAMETER;
        const float max_coord = max(abs(fcoord.x), max(abs(fcoord.y), abs(fcoord.z)));
        const float cascade_float = log2(max_coord / (RCACHE_CASCADE_SIZE / 2 - reserved_cells));
        cascade = uint(clamp(ceil(max(0.0, cascade_float)), 0, RCACHE_CASCADE_COUNT - 1));
    }

    const float cell_diameter = (RCACHE_GRID_CELL_DIAMETER * (1u << cascade));
    const int3 cascade_origin = frame_constants.rcache_cascades[cascade].origin.xyz;

    const float3 normal_based_cell_offset =
        RCACHE_USE_NORMAL_BASED_CELL_OFFSET
        ? normal * cell_diameter * 0.5
        : 0.0.xxx;

    const int3 coord = floor((pos + normal_based_cell_offset) / cell_diameter) - cascade_origin;

    RcacheCoord res;
    res.cascade = cascade;
    res.coord = uint3(clamp(coord, (0).xxx, (RCACHE_CASCADE_SIZE-1).xxx));
    return res;
}

uint rcache_coord_to_cell_idx(RcacheCoord coord) {
    return coord.cell_idx();
}

int3 surfel_pos_to_grid_coord(float3 pos, float3 eye_pos) {
    if (SURFEL_GRID_SCROLL) {
        eye_pos = trunc(eye_pos / RCACHE_GRID_CELL_DIAMETER) * RCACHE_GRID_CELL_DIAMETER;
    } else {
        eye_pos = SURFEL_GRID_CENTER;
    }
    return int3(floor((pos - eye_pos) / RCACHE_GRID_CELL_DIAMETER));
}

float surfel_grid_cell_diameter_in_cascade(uint cascade) {
    return RCACHE_GRID_CELL_DIAMETER * (1u << uint(cascade));
}
