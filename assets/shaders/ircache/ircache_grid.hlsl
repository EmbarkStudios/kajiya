#include "../inc/hash.hlsl"
#include "ircache_constants.hlsl"

// Must match CPU side
static const float IRCACHE_GRID_CELL_DIAMETER = 0.16 * 0.125;

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

uint ws_local_pos_to_cascade_idx(float3 local_pos, uint reserved_cells) {
    const float3 fcoord = local_pos / IRCACHE_GRID_CELL_DIAMETER;
    const float max_coord = max(abs(fcoord.x), max(abs(fcoord.y), abs(fcoord.z)));
    const float cascade_float = log2(max_coord / (IRCACHE_CASCADE_SIZE / 2 - reserved_cells));
    return uint(clamp(ceil(max(0.0, cascade_float)), 0, IRCACHE_CASCADE_COUNT - 1));
}

IrcacheCoord ws_pos_to_ircache_coord(float3 pos, float3 normal, float3 jitter) {
    const float3 center = frame_constants.ircache_grid_center.xyz;

    const uint reserved_cells =
        select(IRCACHE_USE_NORMAL_BASED_CELL_OFFSET
        // Make sure we can actually offset towards the edge of a cascade.
        , 1
        // Business as usual
        , 0);

    float3 cell_offset = 0;

    // Stochastic interpolation (no-op if jitter is zero)
    {
        const uint cascade = ws_local_pos_to_cascade_idx(pos - center, reserved_cells);
        const float cell_diameter = (IRCACHE_GRID_CELL_DIAMETER * (1u << cascade));
        pos += cell_diameter * jitter;
    }

    const uint cascade = ws_local_pos_to_cascade_idx(pos - center, reserved_cells);
    const float cell_diameter = (IRCACHE_GRID_CELL_DIAMETER * (1u << cascade));
    
    const int3 cascade_origin = frame_constants.ircache_cascades[cascade].origin.xyz;

    cell_offset +=
        select(IRCACHE_USE_NORMAL_BASED_CELL_OFFSET
        , normal * cell_diameter * 0.5
        , 0.0.xxx);

    const int3 coord = floor((pos + cell_offset) / cell_diameter) - cascade_origin;

    IrcacheCoord res;
    res.cascade = cascade;
    res.coord = uint3(clamp(coord, (0).xxx, (IRCACHE_CASCADE_SIZE-1).xxx));
    return res;
}

float ircache_grid_cell_diameter_in_cascade(uint cascade) {
    return IRCACHE_GRID_CELL_DIAMETER * (1u << uint(cascade));
}
