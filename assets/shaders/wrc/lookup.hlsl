#ifndef WRC_LOOKUP_HLSL
#define WRC_LOOKUP_HLSL

#include "../inc/pack_unpack.hlsl"

float3 lookup_wrc(int3 probe_coord, float3 dir) {
    const uint probe_idx = probe_coord_to_idx(probe_coord);
    const uint2 tile = wrc_probe_idx_to_atlas_tile(probe_idx);
    const float2 tile_uv = octa_encode(dir);
    const uint2 atlas_px = uint2((tile + tile_uv) * WRC_PROBE_DIMS);
    return wrc_radiance_atlas_tex[atlas_px].rgb;
}

#endif // WRC_LOOKUP_HLSL
