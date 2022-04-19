float3 rtdgi_candidate_ray_dir(uint2 px, float3x3 tangent_to_world) {
    #if 1
        //float2 urand = blue_noise_for_pixel(px, 0).xy;
        //urand = frac(float2(px + urand) / 8.0 + r2_sequence(frame_constants.frame_index));
        float2 urand = blue_noise_for_pixel(px, frame_constants.frame_index).xy;
    #else
        const uint salt = 35110969;
        uint rng = hash3(uint3(px + salt, frame_constants.frame_index));
        const float2 urand = float2(
            uint_to_u01_float(hash1_mut(rng)),
            uint_to_u01_float(hash1_mut(rng))
        );
    #endif
    float3 outgoing_dir;

#if DIFFUSE_GI_BRDF_SAMPLING
    {
        DiffuseBrdf brdf;
        brdf.albedo = 1.0.xxx;
        BrdfSample brdf_sample = brdf.sample(float3(0, 0, 1), urand);
        float3 wi = brdf_sample.wi;
        outgoing_dir = mul(tangent_to_world, wi);
    }
#else
    {
        float3 wi = uniform_sample_hemisphere(urand);
        outgoing_dir = mul(tangent_to_world, wi);
    }
#endif

    return outgoing_dir;
}
