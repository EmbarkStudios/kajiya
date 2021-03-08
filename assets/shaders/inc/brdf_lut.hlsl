#include "samplers.hlsl"
#include "bindless_textures.hlsl"

struct SpecularBrdfEnergyPreservation {
    float3 preintegrated_reflection;
    float3 preintegrated_reflection_mult;
    float3 preintegrated_transmission_fraction;

    static float2 sample_fg_lut(float ndotv, float roughness) {
        float2 uv = float2(ndotv, roughness) * BRDF_FG_LUT_UV_SCALE + BRDF_FG_LUT_UV_BIAS;
        return bindless_textures[0].SampleLevel(sampler_lnc, uv, 0).xy;
    }

    static SpecularBrdfEnergyPreservation from_brdf_ndotv(SpecularBrdf brdf, float ndotv) {
        const float roughness = brdf.roughness;
        const float3 specular_albedo = brdf.albedo;

        float2 fg = sample_fg_lut(ndotv, roughness);
        float3 single_scatter = specular_albedo * fg.x + fg.y;

        #if 0
            // Nothing
            SpecularBrdfEnergyPreservation res;
            res.preintegrated_reflection = single_scatter;
            res.preintegrated_reflection_mult = 1;
            res.preintegrated_transmission_fraction = 1 - res.preintegrated_reflection;
            return res;
        #elif 0
            // Just renormalize
            float3 mult = 1.0 / (fg.x + fg.y);

            SpecularBrdfEnergyPreservation res;
            res.preintegrated_reflection = single_scatter * mult;
            res.preintegrated_reflection_mult = mult;
            res.preintegrated_transmission_fraction = 1 - res.preintegrated_reflection;
            return res;
        #elif 0
            float energy_loss_per_bounce = 1.0 - (fg.x + fg.y);

            // Lost energy accounted for by an infinite geometric series:
            // bounce_radiance = energy_loss_per_bounce * specular_albedo
            /*single_scatter
                + bounce_radiance
                + bounce_radiance * bounce_radiance
                + bounce_radiance * bounce_radiance * bounce_radiance
                + bounce_radiance * bounce_radiance * bounce_radiance * bounce_radiance
                + ...
                ;*/

            float3 bounce_radiance = energy_loss_per_bounce * specular_albedo;
            float3 albedo_inf_series = energy_loss_per_bounce * single_scatter / (1.0 - bounce_radiance);
            float3 corrected = single_scatter + albedo_inf_series;

            SpecularBrdfEnergyPreservation res;
            res.preintegrated_reflection = min(1.0, corrected);
            res.preintegrated_reflection_mult = corrected / max(1e-5, single_scatter);
            res.preintegrated_transmission_fraction = 1 - corrected;
            return res;
        #elif 1
            // The above, reformulated, tweaked
            float Ess = fg.x + fg.y;
            float3 Fss = single_scatter / Ess;
            // Ad-hoc shift towards F90 for subsequent bounces
            float3 Fss2 = lerp(Fss, 1.0, 0.4);
            float3 bounce_radiance = (1.0 - Ess) * Fss2;
            float3 mult = 1.0 + bounce_radiance / (1.0 - bounce_radiance);

            SpecularBrdfEnergyPreservation res;
            res.preintegrated_reflection = single_scatter * mult;
            res.preintegrated_reflection_mult = mult;
            res.preintegrated_transmission_fraction = 1 - res.preintegrated_reflection;
            return res;
        #else
            // https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
            // https://github.com/materialx/MaterialX/blob/a409f8eeae18cc5428154d10d366ecb261094d17/libraries/pbrlib/genglsl/lib/mx_microfacet_specular.glsl#L122

            float Ess = fg.x + fg.y;
            //float3 Fss = specular_albedo;
            float3 Fss = lerp(specular_albedo, 1.0, pow(max(0.0, 1.0 - ndotv), 5));
            //float3 Fss = single_scatter;
            float3 mult = 1.0 + Fss * (1.0 - Ess) / Ess;

            SpecularBrdfEnergyPreservation res;
            res.preintegrated_reflection = single_scatter * mult;
            res.preintegrated_reflection_mult = mult;
            res.preintegrated_transmission_fraction = 1 - res.preintegrated_reflection;
            return res;
        #endif
    }
};
