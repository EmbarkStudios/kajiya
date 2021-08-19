#include "common.hlsl"
#include "../inc/frame_constants.hlsl"

[[vk::binding(0)]] RWTexture3D<float4> direct_tex;
[[vk::binding(1)]] RWTexture3D<float3> indirect_tex;
[[vk::binding(2)]] RWTexture3D<float3> subray_indirect_tex;
[[vk::binding(3)]] cbuffer _ {
    uint cascade_idx;
}

[numthreads(8, 8, 1)]
void main(uint3 dti: SV_DispatchThreadID) {
    const uint dispatch_page_count = CSGI_CARDINAL_DIRECTION_COUNT;
    const uint dispatch_page_idx = dti.x / CSGI_VOLUME_DIMS;

    // The dispatch is `dispatch_page_count` times larger than the volume
    const int3 dispatch_vx = uint3(dti.x % CSGI_VOLUME_DIMS, dti.yz);

    // Ditto, the dispatch is larger than the volume.
    static const uint step = CSGI_VOLUME_DIMS * dispatch_page_count;

    if (csgi_was_dispatch_vx_just_scrolled_in(dispatch_vx, cascade_idx)) {
        // Clear subrays
        #if CSGI_SUBRAY_PACKING_AS_SUBVOLUMES
            // Only works with subvolume subray packing

            {for (uint x = dti.x; x < CSGI_VOLUME_DIMS * CSGI_TOTAL_SUBRAY_COUNT; x += step) {
                subray_indirect_tex[uint3(x, dti.yz)] = 0;
            }}
        #else
            // General, but slower

            {for (uint s = dispatch_page_idx; s < CSGI_CARDINAL_SUBRAY_COUNT; s += dispatch_page_count) {
                for (uint dir_idx = 0; dir_idx < CSGI_CARDINAL_DIRECTION_COUNT; ++dir_idx) {
                    uint3 subray_vx = csgi_cardinal_vx_dir_subray_to_subray_vx(dispatch_vx, dir_idx, s);
                    subray_indirect_tex[subray_vx] = 0;
                }
            }}

            {for (uint s = dispatch_page_idx; s < CSGI_DIAGONAL_SUBRAY_COUNT; s += dispatch_page_count) {
                for (uint dir_idx = CSGI_CARDINAL_DIRECTION_COUNT; dir_idx < CSGI_TOTAL_DIRECTION_COUNT; ++dir_idx) {
                    uint3 subray_vx = csgi_diagonal_vx_dir_subray_to_subray_vx(dispatch_vx, dir_idx, s);
                    subray_indirect_tex[subray_vx] = 0;
                }
            }}
        #endif

        // Clear combined
        {for (uint x = dti.x; x < CSGI_VOLUME_DIMS * CSGI_TOTAL_DIRECTION_COUNT; x += step) {
            indirect_tex[uint3(x, dti.yz)] = 0;
        }}

        direct_tex[dti] = float4(0, 0, 0, 0);
    } else {
        float4 v = direct_tex[dti];

        // Having this branch on makes the sweep passes slower o__O
        // Weird cache behavior?
        //if (any(v > 1e-5))
        {
            direct_tex[dti] = v * (1.0 - CSGI_ACCUM_HYSTERESIS);
        }
    }
}
