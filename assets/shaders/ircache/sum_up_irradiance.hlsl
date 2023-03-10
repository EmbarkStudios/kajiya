#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/sh.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/reservoir.hlsl"
#include "ircache_constants.hlsl"
#include "ircache_sampler_common.inc.hlsl"

[[vk::binding(0)]] StructuredBuffer<uint> ircache_life_buf;
[[vk::binding(1)]] RWByteAddressBuffer ircache_meta_buf;
[[vk::binding(2)]] RWStructuredBuffer<float4> ircache_irradiance_buf;
[[vk::binding(3)]] RWStructuredBuffer<float4> ircache_aux_buf;
[[vk::binding(4)]] StructuredBuffer<uint> ircache_entry_indirection_buf;

struct Contribution {
    float4 sh_rgb[3];

    void add_radiance_in_direction(float3 radiance, float3 direction) {
        // https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/gdc2018-precomputedgiobalilluminationinfrostbite.pdf
        // `shEvaluateL1`, plus the `4` factor, with `pi` cancelled out in the evaluation code (BRDF).
        float4 sh = float4(0.282095, direction * 0.488603) * 4;
        sh_rgb[0] += sh * radiance.r;
        sh_rgb[1] += sh * radiance.g;
        sh_rgb[2] += sh * radiance.b;
    }

    void scale(float value) {
        sh_rgb[0] *= value;
        sh_rgb[1] *= value;
        sh_rgb[2] *= value;
    }
};

[numthreads(64, 1, 1)]
void main(uint dispatch_idx: SV_DispatchThreadID) {
    if (IRCACHE_FREEZE) {
        return;
    }

    const uint total_alloc_count = ircache_meta_buf.Load(IRCACHE_META_TRACING_ALLOC_COUNT);
    if (dispatch_idx >= total_alloc_count) {
        return;
    }

    const uint entry_idx = ircache_entry_indirection_buf[dispatch_idx];
    const uint output_idx = entry_idx * IRCACHE_IRRADIANCE_STRIDE;

    Contribution contribution_sum = (Contribution)0;
    {
        float valid_samples = 0;

        // TODO: counter distortion
        for (uint octa_idx = 0; octa_idx < IRCACHE_OCTA_DIMS2; ++octa_idx) {
            const float2 octa_coord = (float2(octa_idx % IRCACHE_OCTA_DIMS, octa_idx / IRCACHE_OCTA_DIMS) + 0.5) / IRCACHE_OCTA_DIMS;

            const Reservoir1spp r = Reservoir1spp::from_raw(asuint(ircache_aux_buf[entry_idx * IRCACHE_AUX_STRIDE + octa_idx].xy));
            const float3 dir = SampleParams::from_raw(r.payload).direction();

            const float4 contrib = ircache_aux_buf[entry_idx * IRCACHE_AUX_STRIDE + IRCACHE_OCTA_DIMS2 + octa_idx];

            contribution_sum.add_radiance_in_direction(
                contrib.rgb * contrib.w,
                dir
            );

            valid_samples += select(contrib.w > 0, 1.0, 0.0);
        }

        contribution_sum.scale(1.0 / max(1.0, valid_samples));
    }

    for (uint basis_i = 0; basis_i < IRCACHE_IRRADIANCE_STRIDE; ++basis_i) {
        const float4 new_value = contribution_sum.sh_rgb[basis_i];
        float4 prev_value =
            ircache_irradiance_buf[entry_idx * IRCACHE_IRRADIANCE_STRIDE + basis_i]
            * frame_constants.pre_exposure_delta;

        const bool should_reset = all(0.0 == prev_value);
        if (should_reset) {
            prev_value = new_value;
        }

        float blend_factor_new = 0.25;
        //float blend_factor_new = 1;
        const float4 blended_value = lerp(prev_value, new_value, blend_factor_new);

        ircache_irradiance_buf[entry_idx * IRCACHE_IRRADIANCE_STRIDE + basis_i] = blended_value;
    }
}
