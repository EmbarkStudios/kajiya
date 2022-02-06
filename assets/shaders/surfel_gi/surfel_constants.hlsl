#ifndef SURFEL_CONSTANTS_HLSL
#define SURFEL_CONSTANTS_HLSL

#define SURFEL_META_CELL_COUNT (0 * sizeof(uint))
#define SURFEL_META_SURFEL_COUNT (1 * sizeof(uint))
#define SURFEL_META_ALLOC_COUNT (2 * sizeof(uint))

#define SURFEL_LIFE_INVALID 0x8000000

bool is_surfel_life_valid(uint life) {
    //return life < SURFEL_LIFE_INVALID;
    return life < 100;
}

#endif // SURFEL_CONSTANTS_HLSL