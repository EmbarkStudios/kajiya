#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float3> input_tex;
[[vk::binding(1)]] RWTexture3D<float3> output_tex;

[numthreads(4, 4, 4)]
void main(in uint3 vx : SV_DispatchThreadID) {
    uint dir_idx = vx.x / CSGI2_VOLUME_DIMS;

    float3 result = input_tex[vx];
    uint subray_count = dir_idx < 6 ? 4 : 3;

    [unroll]
    for (uint subray = 1; subray < subray_count; ++subray) {
        uint3 subray_offset = uint3(0, subray * CSGI2_VOLUME_DIMS, 0);
        result += input_tex[vx + subray_offset];
    }
    result /= subray_count;

    output_tex[vx] = result;
}
