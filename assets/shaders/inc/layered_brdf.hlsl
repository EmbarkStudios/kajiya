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
        const BrdfValue spec = specular_brdf.evaluate(wo, wi);
        const BrdfValue diff = diffuse_brdf.evaluate(wo, wi);

        return (
            spec.value() * energy_preservation.preintegrated_reflection_mult +
            diff.value() * spec.transmission_fraction
        ) * max(0.0, wi.z);
    }

    BrdfSample sample(float3 wo, float3 urand) {
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
