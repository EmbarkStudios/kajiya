#ifndef IRCACHE_CONSTANTS_HLSL
#define IRCACHE_CONSTANTS_HLSL

#define IRCACHE_USE_TRILINEAR 0
#define IRCACHE_USE_POSITION_VOTING 1
#define IRCACHE_USE_UNIFORM_VOTING 1
#define IRCACHE_FREEZE 0

#define IRCACHE_USE_SPHERICAL_HARMONICS 1

// Same as IRCACHE_META_ALLOC_COUNT, but frozen at the rt dispatch args stage.
#define IRCACHE_META_TRACING_ALLOC_COUNT (0 * sizeof(uint))

#define IRCACHE_META_ENTRY_COUNT (2 * sizeof(uint))
#define IRCACHE_META_ALLOC_COUNT (3 * sizeof(uint))

static const uint IRCACHE_ENTRY_META_OCCUPIED = 1u;
static const uint IRCACHE_ENTRY_META_JUST_ALLOCATED = 2u;

#define IRCACHE_ENTRY_LIFE_RECYCLE 0x8000000u
#define IRCACHE_ENTRY_LIFE_RECYCLED (IRCACHE_ENTRY_LIFE_RECYCLE + 1u)

static const uint IRCACHE_ENTRY_LIFE_PER_RANK = 4;
static const uint IRCACHE_ENTRY_RANK_COUNT = 3;

bool is_ircache_entry_life_valid(uint life) {
    return life < IRCACHE_ENTRY_LIFE_PER_RANK * IRCACHE_ENTRY_RANK_COUNT;
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
static const uint IRCACHE_AUX_STRIDE = 4 * IRCACHE_OCTA_DIMS2;

static const uint IRCACHE_SAMPLES_PER_FRAME = 4;
static const uint IRCACHE_VALIDATION_SAMPLES_PER_FRAME = 4;
static const uint IRCACHE_RESTIR_M_CLAMP = 30;


#endif // IRCACHE_CONSTANTS_HLSL
