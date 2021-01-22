#include "surfel_grid_hash.hlsl"

bool SurfelGridHashEntry::acquire() {
    uint prev_value;
    surfel_hash_key_buf.InterlockedCompareExchange(idx * 4, 0, checksum, prev_value);
    return prev_value == 0;
}
