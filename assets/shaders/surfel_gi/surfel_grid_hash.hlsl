#include "../inc/hash.hlsl"
#include "surfel_constants.hlsl"

static const uint MAX_SURFEL_GRID_CELLS = 1024 * 256;
static const uint SURFEL_CS = 32;
static const float SURFEL_GRID_CASCADE_DIAMETER = SURFEL_GRID_CELL_DIAMETER * SURFEL_CS;
static const float SURFEL_GRID_CASCADE_RADIUS = SURFEL_GRID_CASCADE_DIAMETER * 0.5;

static const float SURFEL_NORMAL_DIRECTION_SQUISH = 2.0;

static const bool SURFEL_GRID_SCROLL = !true;

int3 surfel_pos_to_grid_coord(float3 pos, float3 eye_pos) {
    if (!SURFEL_GRID_SCROLL) {
        eye_pos = 0.0.xxx;
    }
    return int3(floor((pos - eye_pos) / SURFEL_GRID_CELL_DIAMETER));
}

float3 surfel_grid_coord_center(uint4 coord, float3 eye_pos) {
    if (!SURFEL_GRID_SCROLL) {
        eye_pos = 0.0.xxx;
    }
    return eye_pos + ((coord.xyz + 0.5.xxx - SURFEL_CS / 2) * SURFEL_GRID_CELL_DIAMETER) * (1u << uint(coord.w));
}

float surfel_grid_cell_diameter_in_cascade(uint cascade) {
    return SURFEL_GRID_CELL_DIAMETER * (1u << uint(cascade));
}

float surfel_grid_coord_to_cascade_float(int3 coord) {
    const float3 fcoord = coord + 0.5;
    const float max_coord = max(abs(fcoord.x), max(abs(fcoord.y), abs(fcoord.z)));
    return log2(max_coord / (SURFEL_CS / 2));
}

uint surfel_cascade_float_to_cascade(float cascade_float) {
    return uint(clamp(ceil(max(0.0, cascade_float)), 0, 7));
}

uint surfel_grid_coord_to_cascade(int3 coord) {
    return surfel_cascade_float_to_cascade(surfel_grid_coord_to_cascade_float(coord));
}

float surfel_radius_for_pos(float3 pos) {
    return SURFEL_BASE_RADIUS * max(1.0, 
        length(pos - get_eye_position()) / SURFEL_GRID_CASCADE_RADIUS
    );
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

uint surfel_grid_coord_to_checksum(int3 coord) {
    return hash3(asuint(coord));
}

struct SurfelGridHashEntry {
    // True if the entry was found in the hash.
    bool found;

    // If not found, `vacant` will tell whether idx is an empty location
    // at which we can insert a new entry.
    bool vacant;

    // Index into the hash table if found or vacant
    uint idx;

    // Value if found, checksum of the key that was queried if not found.
    //uint checksum;

    // Try to acquire a lock on the vacant entry
    //bool acquire();
};

SurfelGridHashEntry surfel_hash_lookup_by_grid_coord(int3 grid_coord) {
    const uint hash = surfel_grid_coord_to_hash(grid_coord);
    //const uint checksum = surfel_grid_coord_to_hash(grid_coord.zyx);
    //const uint checksum = surfel_grid_coord_to_hash(grid_coord.zyx);

    uint idx = (hash % MAX_SURFEL_GRID_CELLS);

    static const uint MAX_PROBE_COUNT = 1;
    //for (uint i = 0; i < MAX_PROBE_COUNT; ++i, ++idx)
    {
        //const uint entry_checksum = surf_rcache_grid_meta_buf.Load(idx * 4);

        /*if (0 == entry_checksum) {
            SurfelGridHashEntry res;
            res.found = false;
            res.vacant = true;
            res.idx = idx;
            res.checksum = checksum;
            return res;
        }

        if (entry_checksum == checksum)*/
        {
            SurfelGridHashEntry res;
            res.found = true;
            res.vacant = false;
            res.idx = idx;
            //res.checksum = checksum;
            return res;
        }
    }

    /*SurfelGridHashEntry res;
    res.found = false;
    res.vacant = false;
    res.idx = idx;
    res.checksum = checksum;
    return res;*/
}
