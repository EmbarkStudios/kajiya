#include "gbuffer.hlsl"
#include "color.hlsl"

#define LAYERED_BRDF_FORCE_DIFFUSE_ONLY 0
#define LAYERED_BRDF_FORCE_SPECULAR_ONLY 0

// Metalness other than 0.0 and 1.0 loses energy due to the way diffuse albedo
// is spread between the specular and diffuse layers. Scaling both the specular
// and diffuse albedo by a constant can recover this energy.
// This is a reasonably accurate fit (RMSE: 0.0007691) to the scaling function.
float3 metalness_albedo_boost(float metalness, float3 diffuse_albedo) {
    static const float a0 = 1.749;
    static const float a1 = -1.61;
    static const float e1 = 0.5555;
    static const float e3 = 0.8244;

    const float x = metalness;
    const float3 y = diffuse_albedo;
    const float3 y3 = y * y * y;

    return 1.0 + (0.25-(x-0.5)*(x-0.5)) * (a0+a1*abs(x-0.5)) * (e1*y + e3*y3);
}

void apply_metalness_to_brdfs(inout SpecularBrdf specular_brdf, inout DiffuseBrdf diffuse_brdf, float metalness) {
    const float3 albedo = diffuse_brdf.albedo;

    specular_brdf.albedo = lerp(specular_brdf.albedo, albedo, metalness);
    diffuse_brdf.albedo = max(0.0, 1.0 - metalness) * albedo;

    const float3 albedo_boost = metalness_albedo_boost(metalness, albedo);
    specular_brdf.albedo = min(1.0, specular_brdf.albedo * albedo_boost);
    diffuse_brdf.albedo = min(1.0, diffuse_brdf.albedo * albedo_boost);
}

struct LayeredBrdf {
    SpecularBrdf specular_brdf;
    DiffuseBrdf diffuse_brdf;
    SpecularBrdfEnergyPreservation energy_preservation;

    static LayeredBrdf from_gbuffer_ndotv(
        GbufferData gbuffer,
        float ndotv
    ) {
        SpecularBrdf specular_brdf;
        specular_brdf.albedo = 0.04;
        specular_brdf.roughness = gbuffer.roughness;

        DiffuseBrdf diffuse_brdf;
        diffuse_brdf.albedo = gbuffer.albedo;

        apply_metalness_to_brdfs(specular_brdf, diffuse_brdf, gbuffer.metalness);

        LayeredBrdf res;
        res.energy_preservation =
            SpecularBrdfEnergyPreservation::from_brdf_ndotv(specular_brdf, ndotv);

        res.specular_brdf = specular_brdf;
        res.diffuse_brdf = diffuse_brdf;
        return res;
    }

    float3 evaluate(float3 wo, float3 wi) {
        if (wo.z <= 0 || wi.z <= 0) {
            return 0;
        }

        const BrdfValue diff = diffuse_brdf.evaluate(wo, wi);

        #if LAYERED_BRDF_FORCE_DIFFUSE_ONLY
            return diff.value;
        #endif

        const BrdfValue spec = specular_brdf.evaluate(wo, wi);

        #if LAYERED_BRDF_FORCE_SPECULAR_ONLY
            return spec.value;
        #endif

        return (
            spec.value * energy_preservation.preintegrated_reflection_mult +
            diff.value * spec.transmission_fraction
        );
    }

    float3 evaluate_directional_light(float3 wo, float3 wi) {
        if (wo.z <= 0 || wi.z <= 0) {
            return 0;
        }

        const BrdfValue diff = diffuse_brdf.evaluate(wo, wi);

        #if LAYERED_BRDF_FORCE_DIFFUSE_ONLY
            return diff.value;
        #endif

        const BrdfValue spec = specular_brdf.evaluate(wo, wi);

        #if LAYERED_BRDF_FORCE_SPECULAR_ONLY
            return spec.value;
        #endif

        // TODO: multi-scattering on the interface can secondary lobes away from
        // the evaluated direction, which is particularly apparent for directional lights.
        // In the latter case, the following term works better.
        // On the other hand, this will result in energy loss for non-directional lights
        // since the lobes are just redirected, and not lost.
        const float3 preintegrated_reflection_mult_directional =
            //energy_preservation.preintegrated_reflection_mult;
            lerp(1.0, energy_preservation.preintegrated_reflection_mult, sqrt(abs(wi.z)));

        return (
            spec.value * preintegrated_reflection_mult_directional +
            diff.value * spec.transmission_fraction
        );
    }

    BrdfSample sample(float3 wo, float3 urand) {
        #if LAYERED_BRDF_FORCE_DIFFUSE_ONLY
            return diffuse_brdf.sample(wo, urand.xy);
        #endif

        #if LAYERED_BRDF_FORCE_SPECULAR_ONLY
            return specular_brdf.sample(wo, urand.xy);
        #endif

        BrdfSample brdf_sample;

        // We should transmit with throughput equal to `brdf_sample.transmission_fraction`,
        // and reflect with the complement of that. However since we use a single ray,
        // we toss a coin, and choose between reflection and transmission.

        const float spec_wt = calculate_luma(energy_preservation.preintegrated_reflection);
        const float diffuse_wt = calculate_luma(energy_preservation.preintegrated_transmission_fraction * diffuse_brdf.albedo);
        const float transmission_p = diffuse_wt / (spec_wt + diffuse_wt);

        const float lobe_xi = urand.z;
        if (lobe_xi < transmission_p) {
            // Transmission wins! Now sample the bottom layer (diffuse)

            brdf_sample = diffuse_brdf.sample(wo, urand.xy);

            const float lobe_pdf = transmission_p;
            brdf_sample.value_over_pdf /= lobe_pdf;

            // Account for the masking that the top level exerts on the bottom.
            brdf_sample.value_over_pdf *= energy_preservation.preintegrated_transmission_fraction;
        } else {
            // Reflection wins!

            brdf_sample = specular_brdf.sample(wo, urand.xy);

            const float lobe_pdf = (1.0 - transmission_p);
            brdf_sample.value_over_pdf /= lobe_pdf;

            // Apply approximate multi-scatter energy preservation
            brdf_sample.value_over_pdf *= energy_preservation.preintegrated_reflection_mult;
        }

        return brdf_sample;
    }
};
