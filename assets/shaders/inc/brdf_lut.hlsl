#include "samplers.hlsl"
#include "bindless_textures.hlsl"

struct SpecularBrdfEnergyPreservation {
    float3 preintegrated_reflection;
    float3 preintegrated_reflection_mult;
    float3 preintegrated_transmission_fraction;

    static SpecularBrdfEnergyPreservation from_brdf_ndotv(SpecularBrdf brdf, float ndotv) {
        const float roughness = brdf.roughness;
        const float3 specular_albedo = brdf.albedo;

        float2 uv = float2(ndotv, roughness) * BRDF_FG_LUT_UV_SCALE + BRDF_FG_LUT_UV_BIAS;
        float2 fg = bindless_textures[0].SampleLevel(sampler_lnc, uv, 0).xy;

        float3 single_scatter = specular_albedo * fg.x + fg.y;
        float energy_loss_per_bounce = 1.0 - (fg.x + fg.y);

        // Lost energy accounted for by an infinite geometric series:
        /*return single_scatter
            + energy_loss_per_bounce * single_scatter
            + energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * single_scatter
            + energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * single_scatter
            + energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * single_scatter
            + ...
            ;*/

        float3 bounce_radiance = energy_loss_per_bounce * specular_albedo;
        float3 albedo_inf_series = energy_loss_per_bounce * single_scatter / (1.0 - bounce_radiance);
        float3 corrected = single_scatter + albedo_inf_series;

        SpecularBrdfEnergyPreservation res;
        res.preintegrated_reflection = corrected;
        res.preintegrated_reflection_mult = corrected / max(1e-5, single_scatter);
        res.preintegrated_transmission_fraction = 1 - corrected;
        return res;
    }
};
