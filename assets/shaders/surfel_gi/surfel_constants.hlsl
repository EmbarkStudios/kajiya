#ifndef SURFEL_CONSTANTS_HLSL
#define SURFEL_CONSTANTS_HLSL

static const float SURFEL_GRID_CELL_DIAMETER = 0.16;
static const float SURFEL_BASE_RADIUS = SURFEL_GRID_CELL_DIAMETER * 1.2;
static const float SURFEl_RADIUS_OVERSCALE = 1.25;

#define SURF_RCACHE_USE_TRILINEAR 0
#define SURF_RCACHE_USE_UNIFORM_VOTING 1
#define SURF_RCACHE_FREEZE 0

#define SURF_RCACHE_USE_SPHERICAL_HARMONICS 1

//#define SURFEL_META_CELL_COUNT (0 * sizeof(uint))
#define SURFEL_META_ENTRY_COUNT (1 * sizeof(uint))
#define SURFEL_META_ALLOC_COUNT (2 * sizeof(uint))

static const uint SURF_RCACHE_ENTRY_META_OCCUPIED = 1;

#define SURFEL_LIFE_RECYCLE 0x8000000u
#define SURFEL_LIFE_RECYCLED (SURFEL_LIFE_RECYCLE + 1u)

static const uint SURF_RCACHE_ENTRY_LIFE_PER_RANK = 16;
static const uint SURF_RCACHE_ENTRY_RANK_COUNT = 3;

bool is_surfel_life_valid(uint life) {
    return life < SURF_RCACHE_ENTRY_LIFE_PER_RANK * SURF_RCACHE_ENTRY_RANK_COUNT;
}

bool surfel_life_needs_aging(uint life) {
    return life != SURFEL_LIFE_RECYCLED;
}

uint surfel_life_to_rank(uint life) {
    return life / SURF_RCACHE_ENTRY_LIFE_PER_RANK;
}

uint surfel_life_for_rank(uint rank) {
    return rank * SURF_RCACHE_ENTRY_LIFE_PER_RANK;
}

static const uint SURF_RCACHE_OCTA_DIMS = 4;
static const uint SURF_RCACHE_OCTA_DIMS2 = SURF_RCACHE_OCTA_DIMS * SURF_RCACHE_OCTA_DIMS;
static const uint SURF_RCACHE_IRRADIANCE_STRIDE = 3;
static const uint SURF_RCACHE_AUX_STRIDE = 2 * SURF_RCACHE_OCTA_DIMS2;



#endif // SURFEL_CONSTANTS_HLSL