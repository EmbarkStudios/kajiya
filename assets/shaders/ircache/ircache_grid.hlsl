#include "../inc/hash.hlsl"
#include "ircache_constants.hlsl"

static const float IRCACHE_GRID_CELL_DIAMETER = 0.16 * 0.5;
static const uint IRCACHE_CASCADE_SIZE = 32;
static const uint IRCACHE_CASCADE_COUNT = 12;

static const bool IRCACHE_USE_NORMAL_BASED_CELL_OFFSET = true;

struct IrcacheCoord {
    uint3 coord;
    uint cascade;

    static IrcacheCoord from_coord_cascade(uint3 coord, uint cascade) {
        IrcacheCoord res;
        res.coord = min(coord, (IRCACHE_CASCADE_SIZE - 1).xxx);
        res.cascade = min(cascade, IRCACHE_CASCADE_COUNT - 1);
        return res;
    }

    uint cell_idx() {
        return dot(
            uint4(coord, cascade),
            uint4(
                1,
                IRCACHE_CASCADE_SIZE,
                IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE,
                IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE));    
    }
};

IrcacheCoord ws_pos_to_ircache_coord(float3 pos, float3 normal) {
    const float3 center = frame_constants.ircache_grid_center.xyz;

    const uint reserved_cells =
        IRCACHE_USE_NORMAL_BASED_CELL_OFFSET
        // Make sure we can actually offset towards the edge of a cascade.
        ? 1
        // Business as usual
        : 0;

    uint cascade; {
        const float3 fcoord = (pos - center) / IRCACHE_GRID_CELL_DIAMETER;
        const float max_coord = max(abs(fcoord.x), max(abs(fcoord.y), abs(fcoord.z)));
        const float cascade_float = log2(max_coord / (IRCACHE_CASCADE_SIZE / 2 - reserved_cells));
        cascade = uint(clamp(ceil(max(0.0, cascade_float)), 0, IRCACHE_CASCADE_COUNT - 1));
    }

    const float cell_diameter = (IRCACHE_GRID_CELL_DIAMETER * (1u << cascade));
    const int3 cascade_origin = frame_constants.ircache_cascades[cascade].origin.xyz;

    const float3 normal_based_cell_offset =
        IRCACHE_USE_NORMAL_BASED_CELL_OFFSET
        ? normal * cell_diameter * 0.5
        : 0.0.xxx;

    const int3 coord = floor((pos + normal_based_cell_offset) / cell_diameter) - cascade_origin;

    IrcacheCoord res;
    res.cascade = cascade;
    res.coord = uint3(clamp(coord, (0).xxx, (IRCACHE_CASCADE_SIZE-1).xxx));
    return res;
}

float ircache_grid_cell_diameter_in_cascade(uint cascade) {
    return IRCACHE_GRID_CELL_DIAMETER * (1u << uint(cascade));
}
