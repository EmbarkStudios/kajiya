#include "inc/math.hlsl"
#include "inc/quasi_random.hlsl"
#include "inc/samplers.hlsl"
#include "inc/cube_map.hlsl"

[[vk::binding(0)]] TextureCube<float4> input_tex;
[[vk::binding(1)]] RWTexture2DArray<float4> output_tex;
[[vk::binding(2)]] cbuffer _ {
    uint face_width;
}

[numthreads(8, 8, 1)]
void main(in uint3 px : SV_DispatchThreadID) {
    uint face = px.z;
    float2 uv = (px.xy + 0.5) / face_width;

    float3 output_dir = normalize(mul(CUBE_MAP_FACE_ROTATIONS[face], float3(uv * 2 - 1, -1.0)));
    const float3x3 basis = build_orthonormal_basis(output_dir);

    static const uint sample_count = 512;

    uint rng = hash2(px.xy);

    float4 result = 0;
    for (uint i = 0; i < sample_count; ++i) {
        float2 urand = hammersley(i, sample_count);
        float3 input_dir = mul(basis, uniform_sample_cone(urand, 0.99));
        result += input_tex.SampleLevel(sampler_llr, input_dir, 0);
    }

    output_tex[px] = result / sample_count;
}
