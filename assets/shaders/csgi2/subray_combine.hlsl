#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float3> input_tex;
[[vk::binding(1)]] Texture3D<float4> direct_tex;
[[vk::binding(2)]] RWTexture3D<float3> output_tex;

[numthreads(4, 4, 4)]
void main(in uint3 vx : SV_DispatchThreadID) {
    uint dir_idx = vx.x / CSGI2_VOLUME_DIMS;
    float3 indirect_dir = CSGI2_INDIRECT_DIRS[dir_idx];

    float3 result = input_tex[vx];
    uint subray_count = dir_idx < 6 ? 4 : 3;

    uint3 direct_vx = vx % CSGI2_VOLUME_DIMS;
    float opacity = 0;

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
        if (dot(direct_dir, indirect_dir) > 0)
        {
            float a0 = direct_tex[direct_vx + uint3(i * CSGI2_VOLUME_DIMS, 0, 0)].a;
            opacity += a0 * 1;
        }
    }
#endif

    float suppression = 1.0 - saturate(opacity);

    [unroll]
    for (uint subray = 1; subray < subray_count; ++subray) {
        uint3 subray_offset = uint3(0, subray * CSGI2_VOLUME_DIMS, 0);
        result += input_tex[vx + subray_offset];
    }
    result /= subray_count;

    output_tex[vx] = lerp(output_tex[vx], result, 1) * suppression;
}
