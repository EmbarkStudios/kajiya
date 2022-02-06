#ifndef SURFEL_CONSTANTS_HLSL
#define SURFEL_CONSTANTS_HLSL

#define MAX_SURFELS_PER_CELL 128

//#define SURFEL_META_CELL_COUNT (0 * sizeof(uint))
#define SURFEL_META_SURFEL_COUNT (1 * sizeof(uint))
#define SURFEL_META_ALLOC_COUNT (2 * sizeof(uint))

#define SURFEL_LIFE_RECYCLE 0x8000000u
#define SURFEL_LIFE_RECYCLED (SURFEL_LIFE_RECYCLE + 1u)

bool is_surfel_life_valid(uint life) {
    return life < 100;
}

bool surfel_life_needs_aging(uint life) {
    return life != SURFEL_LIFE_RECYCLED;
}

#endif // SURFEL_CONSTANTS_HLSL