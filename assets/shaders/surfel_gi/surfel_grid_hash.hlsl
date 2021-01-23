#include "../inc/hash.hlsl"

// Assumes the existence of the following globals:
// surfel_meta_buf
//      0: uint: number of cells allocated
// surfel_hash_key_buf
//      4*k: checksum

static const uint MAX_SURFEL_GRID_CELLS = 1024 * 1024;
static const float SURFEL_GRID_CELL_DIAMETER = 0.25;

static const float SURFEL_RADIUS = 0.4;
static const float SURFEL_NORMAL_DIRECTION_SQUISH = 2.0;

int3 surfel_pos_to_grid_coord(float3 pos) {
    return int3(floor(pos / SURFEL_GRID_CELL_DIAMETER));
}

float3 surfel_grid_coord_center(int3 coord) {
    return (coord + 0.5.xxx) * SURFEL_GRID_CELL_DIAMETER;
}

uint surfel_grid_coord_to_hash(int3 coord) {
    return hash3(asuint(coord));
}

uint surfel_pos_to_hash(float3 pos) {
    return surfel_grid_coord_to_hash(surfel_pos_to_grid_coord(pos));
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
    uint checksum;

    // Try to acquire a lock on the vacant entry
    bool acquire();
};

SurfelGridHashEntry surfel_hash_lookup_by_grid_coord(int3 grid_coord) {
    const uint hash = surfel_grid_coord_to_hash(grid_coord);
    const uint checksum = surfel_grid_coord_to_hash(grid_coord.zyx);

    uint idx = (hash % MAX_SURFEL_GRID_CELLS);

    static const uint MAX_PROBE_COUNT = 8;
    for (uint i = 0; i < MAX_PROBE_COUNT; ++i, ++idx) {
        const uint entry_checksum = surfel_hash_key_buf.Load(idx * 4);

        if (0 == entry_checksum) {
            SurfelGridHashEntry res;
            res.found = false;
            res.vacant = true;
            res.idx = idx;
            res.checksum = checksum;
            return res;
        }

        if (entry_checksum == checksum) {
            SurfelGridHashEntry res;
            res.found = true;
            res.vacant = false;
            res.idx = idx;
            res.checksum = checksum;
            return res;
        }
    }

    SurfelGridHashEntry res;
    res.found = false;
    res.vacant = false;
    res.idx = idx;
    res.checksum = 0;
    return res;
}
