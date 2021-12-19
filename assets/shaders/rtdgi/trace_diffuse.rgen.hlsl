#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../csgi/common.hlsl"

// Should be 1, but rarely matters for the diffuse bounce, so might as well save a few cycles.
#define USE_SOFT_SHADOWS 0

#define USE_CSGI 1

// Experimental. Better precision, but also higher variance because of unfiltered lookups.
#define USE_CSGI_SUBRAYS 0

#define USE_TEMPORAL_JITTER 1
#define USE_SHORT_RAYS_ONLY 1
#define SHORT_RAY_SIZE_VOXEL_CELLS 4.0
#define ROUGHNESS_BIAS 0.5
#define SUPPRESS_GI_FOR_NEAR_HITS 1
#define USE_SCREEN_GI_REPROJECTION 1

#define USE_EMISSIVE 1
#define USE_LIGHTS 1

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> reprojected_gi_tex;
[[vk::binding(3)]] Texture2D<float> ssao_tex;
DEFINE_BLUE_NOISE_SAMPLER_BINDINGS(4, 5, 6)
[[vk::binding(7)]] RWTexture2D<float4> out0_tex;
[[vk::binding(8)]] Texture3D<float4> csgi_indirect_tex[CSGI_CASCADE_COUNT];
[[vk::binding(9)]] Texture3D<float3> csgi_subray_indirect_tex[CSGI_CASCADE_COUNT];
[[vk::binding(10)]] Texture3D<float> csgi_opacity_tex[CSGI_CASCADE_COUNT];
[[vk::binding(11)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(12)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

#include "../csgi/lookup.hlsl"
#include "../csgi/subray_lookup.hlsl"

static const float SKY_DIST = 1e4;

[shader("raygeneration")]
void main() {
    uint2 hi_px_subpixels[4] = {
        uint2(0, 0),
        uint2(1, 1),
        uint2(1, 0),
        uint2(0, 1),
    };

    const uint2 px = DispatchRaysIndex().xy;
    const uint2 hi_px = px * 2 + hi_px_subpixels[frame_constants.frame_index & 3];
    
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        out0_tex[px] = float4(0.0.xxx, -SKY_DIST);
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    float4 gbuffer_packed = gbuffer_tex[hi_px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();

    const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
    const float3 refl_ray_origin = view_ray_context.biased_secondary_ray_origin_ws();

    float3 wo = mul(-view_ray_context.ray_dir_ws(), tangent_to_world);
    const float3 primary_hit_normal = gbuffer.normal;

    // Hack for shading normals facing away from the outgoing ray's direction:
    // We flip the outgoing ray along the shading normal, so that the reflection's curvature
    // continues, albeit at a lower rate.
    if (wo.z < 0.0) {
        wo.z *= -0.25;
        wo = normalize(wo);
    }

    DiffuseBrdf brdf;
    brdf.albedo = 1.0.xxx;

    const uint seed = USE_TEMPORAL_JITTER ? frame_constants.frame_index : 0;
    uint rng = hash3(uint3(px, seed));

#if 0
    const uint noise_offset = frame_constants.frame_index * (USE_TEMPORAL_JITTER ? 1 : 0);

    float2 urand = float2(
        blue_noise_sampler(px.x, px.y, noise_offset, 0),
        blue_noise_sampler(px.x, px.y, noise_offset, 1)
    );
#elif 1
    // 256x256 blue noise

    const uint noise_offset = frame_constants.frame_index * (USE_TEMPORAL_JITTER ? 1 : 0);
    float2 urand = blue_noise_for_pixel(px, noise_offset).xy;
#elif 1
    float2 urand = float2(
        uint_to_u01_float(hash1_mut(rng)),
        uint_to_u01_float(hash1_mut(rng))
    );
#else
    float2 urand = frac(
        hammersley((frame_constants.frame_index * 5) % 16, 16) +
        bindless_textures[BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0][px & 255].xy * 255.0 / 256.0 + 0.5 / 256.0
    );
#endif

    float3 total_radiance = 0.0.xxx;

    // HACK; should be in dedicated passes
    if (USE_LIGHTS) {
        float2 urand = blue_noise_for_pixel(px, frame_constants.frame_index + 100).xy;

        for (uint light_idx = 0; light_idx < frame_constants.triangle_light_count; light_idx += 1) {
            TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
            LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
            const float3 shadow_ray_origin = view_ray_context.ray_hit_ws();
            const float3 to_light_ws = light_sample.pos - shadow_ray_origin;
            const float dist_to_light2 = dot(to_light_ws, to_light_ws);
            const float3 to_light_norm_ws = to_light_ws * rsqrt(dist_to_light2);

            const float to_psa_metric =
                max(0.0, dot(to_light_norm_ws, gbuffer.normal))
                * max(0.0, dot(to_light_norm_ws, -light_sample.normal))
                / dist_to_light2;

            if (to_psa_metric > 0.0) {
                const bool is_shadowed =
                    rt_is_shadowed(
                        acceleration_structure,
                        new_ray(
                            shadow_ray_origin,
                            to_light_norm_ws,
                            1e-3,
                            sqrt(dist_to_light2) - 2e-3
                    ));

                total_radiance +=
                    !is_shadowed ? (triangle_light.radiance() * brdf.albedo / light_sample.pdf.value * to_psa_metric / M_PI) : 0;
            }
        }
    }

    BrdfSample brdf_sample = brdf.sample(wo, urand);

    //const float origin_cascade_idx = csgi_blended_cascade_idx_for_pos(refl_ray_origin);
    const float origin_cascade_idx = csgi_cascade_idx_for_pos(refl_ray_origin);

    if (brdf_sample.is_valid()) {
        RayDesc outgoing_ray;
        outgoing_ray.Direction = mul(tangent_to_world, brdf_sample.wi);
        outgoing_ray.Origin = refl_ray_origin;
        outgoing_ray.TMin = 0;

        #if USE_SHORT_RAYS_ONLY
            outgoing_ray.TMax = csgi_blended_voxel_size(origin_cascade_idx).x * SHORT_RAY_SIZE_VOXEL_CELLS;
        #else
            outgoing_ray.TMax = SKY_DIST;
        #endif

        // If the ray goes from a higher-res cascade to a lower-res one, it might end up
        // terminating too early. Re-calculate the max trace range based on where we'd finish.
        #if USE_SHORT_RAYS_ONLY && CSGI_CASCADE_COUNT > 1
            uint end_cascade_idx = csgi_cascade_idx_for_pos(
                outgoing_ray.Direction + outgoing_ray.Origin * outgoing_ray.TMax
            );
            outgoing_ray.TMax = max(
                outgoing_ray.TMax,
                csgi_voxel_size(end_cascade_idx).x * SHORT_RAY_SIZE_VOXEL_CELLS
            );
        #endif

        // The control variates used in the temporal filter are based on a regular CSGI lookup.
        // For proper integration, that GI lookup should cancel out with the control variate value
        // used in this pass.
        // Our control variate formulation is:
        //
        // ∫ (precise_gi(x, w) - csgi_directional(x, w)) - csgi(x)
        //
        // The assumption being that ∫ csgi_directional(x, w) == csgi(x)
        //
        // While most of the time this works, that assumption is not correct because CSGI
        // is not integrated exactly the same way as the hemispherical integration in this function.
        // Errors tend to pop up in corners and in areas of tricky visibility. In that case,
        // leaks and darkening can appear in lighting.
        //
        // It is then better to switch to the (ineffective) formulation:
        // ∫ (precise_gi(x, w) - csgi(x)) - csgi(x)
        //
        // This does not provide any benefits for variance reduction, but it eliminates the artifacts.
        bool control_variate_sample_directional = ssao_tex[hi_px] > 0.8;

        const float reflected_cone_spread_angle = 0.2;
        const RayCone ray_cone =
            pixel_ray_cone_from_image_height(gbuffer_tex_size.y * 0.5)
            .propagate(reflected_cone_spread_angle, length(outgoing_ray.Origin - get_eye_position()));

        const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
            .with_cone(ray_cone)
            .with_cull_back_faces(true)
            .with_path_length(1)
            .trace(acceleration_structure);

        if (primary_hit.is_hit) {
            GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();

            // Project the sample into clip space, and check if it's on-screen
            const float3 primary_hit_cs = position_world_to_clip(primary_hit.position);
            const float2 primary_hit_uv = cs_to_uv(primary_hit_cs.xy);
            const float primary_hit_screen_depth = depth_tex.SampleLevel(sampler_nnc, primary_hit_uv, 0);
            const GbufferDataPacked primary_hit_screen_gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex.SampleLevel(sampler_nnc, primary_hit_uv, 0)));
            const float3 primary_hit_screen_normal_ws = primary_hit_screen_gbuffer.unpack_normal();
            bool is_on_screen =
                all(abs(primary_hit_cs.xy) < 1.0)
                && inverse_depth_relative_diff(primary_hit_cs.z, primary_hit_screen_depth) < 5e-3
                && dot(primary_hit_screen_normal_ws, -outgoing_ray.Direction) > 0.0
                && dot(primary_hit_screen_normal_ws, gbuffer.normal) > 0.7
                ;

            // If it is on-screen, we'll try to use its reprojected radiance from the previous frame
            float4 reprojected_radiance = 0;
            if (is_on_screen) {
                reprojected_radiance =
                    reprojected_gi_tex.SampleLevel(sampler_nnc, primary_hit_uv, 0);

                // Check if the temporal reprojection is valid.
                is_on_screen = reprojected_radiance.w > 0;
            }

            gbuffer.roughness = lerp(gbuffer.roughness, 1.0, ROUGHNESS_BIAS);
            const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
            const float3 wo = mul(-outgoing_ray.Direction, tangent_to_world);
            const LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

            // Sun
            float3 sun_radiance = SUN_COLOR;
            if (any(sun_radiance) > 0) {
                const float3 to_light_norm = sample_sun_direction(
                    blue_noise_for_pixel(px, frame_constants.frame_index).xy,
                    USE_SOFT_SHADOWS
                );

                const bool is_shadowed =
                    rt_is_shadowed(
                        acceleration_structure,
                        new_ray(
                            primary_hit.position,
                            to_light_norm,
                            1e-4,
                            SKY_DIST
                    ));

                const float3 wi = mul(to_light_norm, tangent_to_world);
                const float3 brdf_value = brdf.evaluate(wo, wi) * max(0.0, wi.z);
                const float3 light_radiance = is_shadowed ? 0.0 : sun_radiance;
                total_radiance += brdf_value * light_radiance;
            }

            if (USE_EMISSIVE) {
                total_radiance += gbuffer.emissive;
            }

            if (USE_SCREEN_GI_REPROJECTION && is_on_screen) {
                total_radiance += reprojected_radiance.rgb * gbuffer.albedo;
            } else {
                if (USE_LIGHTS) {
                    float2 urand = float2(
                        uint_to_u01_float(hash1_mut(rng)),
                        uint_to_u01_float(hash1_mut(rng))
                    );

                    for (uint light_idx = 0; light_idx < frame_constants.triangle_light_count; light_idx += 1) {
                        TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
                        LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
                        const float3 shadow_ray_origin = primary_hit.position;
                        const float3 to_light_ws = light_sample.pos - shadow_ray_origin;
                        const float dist_to_light2 = dot(to_light_ws, to_light_ws);
                        const float3 to_light_norm_ws = to_light_ws * rsqrt(dist_to_light2);

                        const float to_psa_metric =
                            max(0.0, dot(to_light_norm_ws, gbuffer.normal))
                            * max(0.0, dot(to_light_norm_ws, -light_sample.normal))
                            / dist_to_light2;

                        if (to_psa_metric > 0.0) {
                            const bool is_shadowed =
                                rt_is_shadowed(
                                    acceleration_structure,
                                    new_ray(
                                        shadow_ray_origin,
                                        to_light_norm_ws,
                                        1e-3,
                                        sqrt(dist_to_light2) - 2e-3
                                ));

                            #if 1
                                const float3 bounce_albedo = lerp(gbuffer.albedo, 1.0.xxx, 0.04);
                                const float3 brdf_value = bounce_albedo * to_psa_metric / M_PI;
                            #else
                                const float3 wi = mul(to_light_norm_ws, tangent_to_world);
                                const float3 brdf_value = brdf.evaluate(wo, wi) * to_psa_metric;
                            #endif

                            total_radiance +=
                                !is_shadowed ? (triangle_light.radiance() * brdf_value / light_sample.pdf.value) : 0;
                        }
                    }
                }

                if (USE_CSGI) {
                    const float3 pseudo_bent_normal = normalize(normalize(get_eye_position() - primary_hit.position) + gbuffer.normal);

                    CsgiLookupParams lookup_params =
                        CsgiLookupParams::make_default()
                            .with_bent_normal(pseudo_bent_normal)
                            ;

                    // doesn't seem to change much from using origin_cascade_idx
                    //const uint hit_cascade_idx = csgi_cascade_idx_for_pos(primary_hit.position);

                    if (SUPPRESS_GI_FOR_NEAR_HITS && primary_hit.ray_t <= csgi_blended_voxel_size(origin_cascade_idx).x) {
                        float max_normal_offset = primary_hit.ray_t * abs(dot(outgoing_ray.Direction, gbuffer.normal));

                        // Suppression in open corners causes excessive darkening,
                        // and doesn't prevent that many leaks. This strikes a balance.
                        const float normal_agreement = dot(primary_hit_normal, gbuffer.normal);
                        max_normal_offset = lerp(max_normal_offset, 1.51, normal_agreement * 0.5 + 0.5);

                        lookup_params = lookup_params
                            .with_max_normal_offset_scale(max_normal_offset / csgi_blended_voxel_size(origin_cascade_idx).x);

    					control_variate_sample_directional = false;
                    }

                    float3 csgi = lookup_csgi(
                        primary_hit.position,
                        gbuffer.normal,
                        lookup_params
                    );

                    total_radiance += csgi * gbuffer.albedo;
                }
            }
        } else {
            #if USE_SHORT_RAYS_ONLY
                const float3 csgi_lookup_pos = outgoing_ray.Origin + outgoing_ray.Direction * max(0.0, outgoing_ray.TMax - csgi_blended_voxel_size(origin_cascade_idx).x);

                #if USE_CSGI_SUBRAYS
                    float3 subray_contrib = point_sample_csgi_subray_indirect(csgi_lookup_pos, outgoing_ray.Direction);
                    {
                        uint lookup_cascade_idx = csgi_cascade_idx_for_pos(csgi_lookup_pos);
                        const float3 vol_pos = (csgi_lookup_pos - CSGI_VOLUME_ORIGIN);
                        int3 gi_vx = int3(floor(vol_pos / csgi_voxel_size(lookup_cascade_idx)));
                        uint3 vx = csgi_wrap_vx_within_cascade(gi_vx);

                        total_radiance += subray_contrib * smoothstep(0.5, 1, csgi_opacity_tex[lookup_cascade_idx][vx]);
                    }
                #else
                    total_radiance += lookup_csgi(
                        csgi_lookup_pos,
	                    0.0.xxx,    // don't offset by any normal
    	                CsgiLookupParams::make_default()
        	                .with_sample_directional_radiance(outgoing_ray.Direction)
                );
                #endif
            #else
                total_radiance += sky_cube_tex.SampleLevel(sampler_llr, outgoing_ray.Direction, 0).rgb;
            #endif
        }

        float3 control_variate = 0.0.xxx;
        {
            float3 to_eye = get_eye_position() - view_ray_context.ray_hit_ws();
            float3 pseudo_bent_normal = normalize(normalize(to_eye) + gbuffer.normal);

            CsgiLookupParams lookup_params = CsgiLookupParams::make_default()
                .with_bent_normal(pseudo_bent_normal)
                ;

            if (control_variate_sample_directional) {
                lookup_params = lookup_params
                    .with_sample_directional_radiance(outgoing_ray.Direction);
            }

            control_variate = lookup_csgi(
                view_ray_context.ray_hit_ws(),
                gbuffer.normal,
                lookup_params
            );
        }

        #if USE_RTDGI_CONTROL_VARIATES
            float3 out_value = total_radiance - control_variate;
            //float3 out_value = control_variate;
        #else
            float3 out_value = total_radiance;
        #endif
        //float3 out_value = control_variate;

        out0_tex[px] = float4(out_value, 1);
    } else {
        out0_tex[px] = float4(0.0.xxx, 1);
    }
}
