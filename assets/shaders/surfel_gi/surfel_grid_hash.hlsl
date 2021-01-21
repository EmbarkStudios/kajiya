// Assumes the existence of the following globals:
// surfel_meta_buf
//      0: uint: number of cells allocated
// surfel_hash_key_buf
//      4*k: checksum
// surfel_hash_value_buf
//      4*k: value

static const uint MAX_SURFEL_GRID_CELLS = 1024 * 1024;

uint surfel_pos_to_hash(float3 pos) {
    return hash3(asuint(int3(floor(pos * 3.0))));
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
    bool acquire() {
        uint prev_value;
        surfel_hash_key_buf.InterlockedCompareExchange(idx * 4, 0, checksum, prev_value);
        return prev_value == 0;
    }
};

SurfelGridHashEntry surfel_hash_lookup(float3 pos) {
    const uint hash = surfel_pos_to_hash(pos);
    const uint checksum = surfel_pos_to_hash(pos.zyx);

    uint idx = (hash % MAX_SURFEL_GRID_CELLS);

    static const uint MAX_PROBE_COUNT = 4;
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
