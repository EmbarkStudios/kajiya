#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float3> input_tex;
[[vk::binding(1)]] Texture3D<float4> direct_tex;
[[vk::binding(2)]] RWTexture3D<float3> output_tex;

#define CLAMP_STRATEGY_NONE 0
#define CLAMP_STRATEGY_STRONG_DIRECTIONAL 1
#define CLAMP_STRATEGY_WEAK_DIRECTIONAL 2
#define CLAMP_STRATEGY_WEAK_OMNI 3
#define CLAMP_STRATEGY_STRONG_OMNI 1

#define CLAMP_STRATEGY CLAMP_STRATEGY_WEAK_OMNI
//#define CLAMP_STRATEGY CLAMP_STRATEGY_NONE

#if CLAMP_STRATEGY == CLAMP_STRATEGY_NONE
    #define USE_INDIRECT_CLAMP 0
    #define INDIRECT_CLAMP_DIRECTIONAL 0
    #define INDIRECT_CLAMP_AMOUNT 0
#elif CLAMP_STRATEGY == CLAMP_STRATEGY_STRONG_DIRECTIONAL
    #define USE_INDIRECT_CLAMP 1
    #define INDIRECT_CLAMP_DIRECTIONAL 1
    #define INDIRECT_CLAMP_AMOUNT 1.0
#elif CLAMP_STRATEGY == CLAMP_STRATEGY_WEAK_DIRECTIONAL
    #define USE_INDIRECT_CLAMP 1
    #define INDIRECT_CLAMP_DIRECTIONAL 1
    #define INDIRECT_CLAMP_AMOUNT 0.5
#elif CLAMP_STRATEGY == CLAMP_STRATEGY_WEAK_OMNI
    #define USE_INDIRECT_CLAMP 1
    #define INDIRECT_CLAMP_DIRECTIONAL 0
    #define INDIRECT_CLAMP_AMOUNT 0.25
#elif CLAMP_STRATEGY == CLAMP_STRATEGY_STRONG_OMNI
    #define USE_INDIRECT_CLAMP 1
    #define INDIRECT_CLAMP_DIRECTIONAL 0
    #define INDIRECT_CLAMP_AMOUNT 1.0
#endif

[numthreads(4, 4, 4)]
void main(in uint3 vx : SV_DispatchThreadID) {
    uint dir_idx = vx.x / CSGI2_VOLUME_DIMS;
    float3 indirect_dir = CSGI2_INDIRECT_DIRS[dir_idx];

    float3 result = input_tex[vx];
    uint subray_count = dir_idx < 6 ? 4 : 3;

    uint3 direct_vx = vx % CSGI2_VOLUME_DIMS;
    float opacity = 0;

    #if USE_INDIRECT_CLAMP
        #if 0
            for (uint i = 0; i < 6; i += 2) {
                float3 direct_dir = CSGI2_SLICE_DIRS[i];

                // Only block directions that share axes
                if (abs(dot(direct_dir, indirect_dir)) > 0)
                {
                    float a0 = direct_tex[direct_vx + uint3(i * CSGI2_VOLUME_DIMS, 0, 0)].a;
                    float a1 = direct_tex[direct_vx + uint3((i+1) * CSGI2_VOLUME_DIMS, 0, 0)].a;
                    //opacity += 100 * a0 * a1;    // Only block conflicting directions; LEAKS
                    opacity += a0 + a1;
                }
            }
        #else
            for (uint i = 0; i < 6; ++i) {
                float3 direct_dir = CSGI2_SLICE_DIRS[i];

                // Only block relevant directions
                if (!INDIRECT_CLAMP_DIRECTIONAL || dot(direct_dir, indirect_dir) > 0) {
                    float a0 = direct_tex[direct_vx + uint3(i * CSGI2_VOLUME_DIMS, 0, 0)].a;
                    opacity += a0 * 1.0;
                }
            }
        #endif
    #endif

    float suppression = 1.0 - saturate(opacity * INDIRECT_CLAMP_AMOUNT);

    [unroll]
    for (uint subray = 1; subray < subray_count; ++subray) {
        uint3 subray_offset = uint3(0, subray * CSGI2_VOLUME_DIMS, 0);
        result += input_tex[vx + subray_offset];
    }
    result /= subray_count;

    output_tex[vx] = lerp(output_tex[vx], result, 1) * suppression;
}
