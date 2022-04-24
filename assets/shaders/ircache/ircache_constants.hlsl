#ifndef IRCACHE_CONSTANTS_HLSL
#define IRCACHE_CONSTANTS_HLSL

#define IRCACHE_USE_TRILINEAR 0
#define IRCACHE_USE_UNIFORM_VOTING 1
#define IRCACHE_FREEZE 0

#define IRCACHE_USE_SPHERICAL_HARMONICS 1

//#define IRCACHE_META_CELL_COUNT (0 * sizeof(uint))
#define IRCACHE_META_ENTRY_COUNT (1 * sizeof(uint))
#define IRCACHE_META_ALLOC_COUNT (2 * sizeof(uint))

static const uint IRCACHE_ENTRY_META_OCCUPIED = 1;

#define IRCACHE_ENTRY_LIFE_RECYCLE 0x8000000u
#define IRCACHE_ENTRY_LIFE_RECYCLED (IRCACHE_ENTRY_LIFE_RECYCLE + 1u)

static const uint IRCACHE_ENTRY_LIFE_PER_RANK = 16;
static const uint IRCACHE_ENTRY_RANK_COUNT = 3;

bool is_ircache_entry_life_valid(uint life) {
    return life < IRCACHE_ENTRY_LIFE_PER_RANK * IRCACHE_ENTRY_RANK_COUNT;
}

bool ircache_entry_life_needs_aging(uint life) {
    return life != IRCACHE_ENTRY_LIFE_RECYCLED;
}

uint ircache_entry_life_to_rank(uint life) {
    return life / IRCACHE_ENTRY_LIFE_PER_RANK;
}

uint ircache_entry_life_for_rank(uint rank) {
    return rank * IRCACHE_ENTRY_LIFE_PER_RANK;
}

static const uint IRCACHE_OCTA_DIMS = 4;
static const uint IRCACHE_OCTA_DIMS2 = IRCACHE_OCTA_DIMS * IRCACHE_OCTA_DIMS;
static const uint IRCACHE_IRRADIANCE_STRIDE = 3;
static const uint IRCACHE_AUX_STRIDE = 2 * IRCACHE_OCTA_DIMS2;



#endif // IRCACHE_CONSTANTS_HLSL