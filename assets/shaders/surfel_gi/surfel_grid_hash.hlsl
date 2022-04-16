#include "../inc/hash.hlsl"
#include "surfel_constants.hlsl"

static const uint MAX_SURFEL_GRID_CELLS = 1024 * 1024 * 2;
static const uint SURFEL_CS = 32;

static const bool SURFEL_GRID_SCROLL = true;
//static const float3 SURFEL_GRID_CENTER = float3(-1.5570648, -1.2360737, 9.283543);
static const float3 SURFEL_GRID_CENTER = float3(0, 0, 14);
//static const float3 SURFEL_GRID_CENTER = float3(-2, 0, -2);
//static const float3 SURFEL_GRID_CENTER = float3(-1.3989427, 0.44028947, -4.080884);
//static const float3 SURFEL_GRID_CENTER = 0.0.xxx;

struct RcacheCoord {
    uint3 coord;
    uint cascade;
};

/*RcacheCoord ws_pos_to_rcache_coord(float3 pos) {

}*/

int3 surfel_pos_to_grid_coord(float3 pos, float3 eye_pos) {
    if (SURFEL_GRID_SCROLL) {
        eye_pos = trunc(eye_pos / SURFEL_GRID_CELL_DIAMETER) * SURFEL_GRID_CELL_DIAMETER;
    } else {
        eye_pos = SURFEL_GRID_CENTER;
    }
    return int3(floor((pos - eye_pos) / SURFEL_GRID_CELL_DIAMETER));
}

float surfel_grid_cell_diameter_in_cascade(uint cascade) {
    return SURFEL_GRID_CELL_DIAMETER * (1u << uint(cascade));
}

uint surfel_grid_coord_to_cascade(int3 coord) {
    const float3 fcoord = coord + 0.5;
    const float max_coord = max(abs(fcoord.x), max(abs(fcoord.y), abs(fcoord.z)));
    const float cascade_float = log2(max_coord / (SURFEL_CS / 2));
    return uint(clamp(ceil(max(0.0, cascade_float)), 0, 7));
}

int3 surfel_grid_coord_within_cascade(int3 coord, uint cascade) {
    //return coord / int(1u << cascade) + SURFEL_CS / 2;
    //return (coord + ((SURFEL_CS / 2) << cascade)) / (1u << cascade);
    //return (coord + ((SURFEL_CS / 2) << cascade)) >> cascade;

    return (coord >> cascade) + SURFEL_CS / 2;
}

uint4 surfel_grid_coord_to_c4(int3 coord) {
    const uint cascade = surfel_grid_coord_to_cascade(coord);
    const uint3 ucoord_in_cascade = clamp(surfel_grid_coord_within_cascade(coord, cascade), (int3)0, (int3)(SURFEL_CS - 1));
    //const uint3 ucoord_in_cascade = max(0, surfel_grid_coord_within_cascade(coord, cascade));
    return uint4(ucoord_in_cascade, cascade);
}

uint surfel_grid_c4_to_hash(uint4 c4) {
    return dot(
        c4,
        uint4(
            1,
            SURFEL_CS,
            SURFEL_CS * SURFEL_CS,
            SURFEL_CS * SURFEL_CS * SURFEL_CS));
}

uint surfel_grid_coord_to_hash(int3 coord) {
    return surfel_grid_c4_to_hash(surfel_grid_coord_to_c4(coord));
}
