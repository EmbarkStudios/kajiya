float3 rtdgi_candidate_ray_dir(uint2 px, float3x3 tangent_to_world) {
    float2 urand = blue_noise_for_pixel(px, 0).xy;
    urand = frac(float2(px + urand) / 8.0 + r2_sequence(frame_constants.frame_index));
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
        //float3 wi = uniform_sample_hemisphere(urand);
        float3 od; {
            od = uniform_sample_sphere(urand);
        }
        outgoing_dir = od;
    }
#endif

    return outgoing_dir;
}
