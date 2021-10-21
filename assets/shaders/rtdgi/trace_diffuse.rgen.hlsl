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
#include "../inc/reservoir.hlsl"
#include "../csgi/common.hlsl"
#include "restir_settings.hlsl"

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
[[vk::binding(7)]] Texture2D<float4> irradiance_history_tex;
[[vk::binding(8)]] Texture2D<float4> ray_history_tex;
[[vk::binding(9)]] Texture2D<float4> reservoir_history_tex;
[[vk::binding(10)]] Texture2D<float4> reprojection_tex;
[[vk::binding(11)]] RWTexture2D<float4> irradiance_out_tex;
[[vk::binding(12)]] RWTexture2D<float4> ray_out_tex;
[[vk::binding(13)]] RWTexture2D<float4> hit0_out_tex;
[[vk::binding(14)]] RWTexture2D<float4> reservoir_out_tex;
[[vk::binding(15)]] Texture3D<float4> csgi_indirect_tex[CSGI_CASCADE_COUNT];
[[vk::binding(16)]] Texture3D<float3> csgi_subray_indirect_tex[CSGI_CASCADE_COUNT];
[[vk::binding(17)]] Texture3D<float> csgi_opacity_tex[CSGI_CASCADE_COUNT];
[[vk::binding(18)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(19)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

#include "../csgi/lookup.hlsl"
#include "../csgi/subray_lookup.hlsl"

static const float SKY_DIST = 1e5;

struct TraceResult {
    float3 out_value;
    float hit_t;
};

TraceResult do_the_thing(uint2 px, inout uint rng, RayDesc outgoing_ray, GbufferData gbuffer) {
    const float3 primary_hit_normal = gbuffer.normal;
    //const float origin_cascade_idx = csgi_blended_cascade_idx_for_pos(refl_ray_origin);
    const float origin_cascade_idx = csgi_cascade_idx_for_pos(outgoing_ray.Origin);
    /*if (origin_cascade_idx > 0) {
        irradiance_out_tex[px] = float4(0.0.xxx, -SKY_DIST);
        return;
    }*/

    float3 total_radiance = 0.0.xxx;

    #if USE_SHORT_RAYS_ONLY
        outgoing_ray.TMax = csgi_blended_voxel_size(origin_cascade_idx).x * SHORT_RAY_SIZE_VOXEL_CELLS;
    #else
        outgoing_ray.TMax = SKY_DIST;
    #endif

    float hit_t = outgoing_ray.TMax;

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

    // TODO: cone spread angle
    const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
        .with_cone_width(0.05)
        .with_cull_back_faces(true)
        .with_path_length(1)
        .trace(acceleration_structure);

    if (primary_hit.is_hit) {
        hit_t = primary_hit.ray_t;
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
                        //.with_linear_fetch(false)
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
                }

                float3 csgi = lookup_csgi(
                    primary_hit.position,
                    gbuffer.normal,
                    lookup_params
                );

                //if (primary_hit.ray_t > csgi_voxel_size(origin_cascade_idx).x)
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
                        //.with_directional_radiance_phong_exponent(8)
            );
            #endif
        #else
            total_radiance += sky_cube_tex.SampleLevel(sampler_llr, outgoing_ray.Direction, 0).rgb;
        #endif
    }

    float3 out_value = total_radiance;

    //out_value /= reservoir.p_sel;

    TraceResult result;
    result.out_value = out_value;
    result.hit_t = hit_t;
    return result;
}

[shader("raygeneration")]
void main() {
    const uint2 hi_px_subpixels[4] = {
        uint2(0, 0),
        uint2(1, 1),
        uint2(1, 0),
        uint2(0, 1),
    };

    const uint2 px = DispatchRaysIndex().xy;
    const uint2 hi_px = px * 2 + hi_px_subpixels[frame_constants.frame_index & 3];
    
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        irradiance_out_tex[px] = float4(0.0.xxx, -SKY_DIST);
        hit0_out_tex[px] = 0.0.xxxx;
        reservoir_out_tex[px] = 0.0.xxxx;
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    float4 gbuffer_packed = gbuffer_tex[hi_px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();

    const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
    const float3 refl_ray_origin = view_ray_context.biased_secondary_ray_origin_ws();

    float3 wo = mul(-view_ray_context.ray_dir_ws(), tangent_to_world);

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

    // TODO: use
    float3 light_radiance = 0.0.xxx;

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

                light_radiance +=
                    !is_shadowed ? (triangle_light.radiance() * brdf.albedo / light_sample.pdf.value * to_psa_metric / M_PI) : 0;
            }
        }
    }

    BrdfSample brdf_sample = brdf.sample(wo, urand);

    if (!brdf_sample.is_valid()) {
        irradiance_out_tex[px] = float4(0.0.xxx, 1);
        ray_out_tex[px] = 0.0.xxxx;
        hit0_out_tex[px] = 0.0.xxxx;
        reservoir_out_tex[px] = 0.0.xxxx;
        return;
    }

    float3 outgoing_dir = gbuffer.normal;
    float p_q_sel = 0;

    Reservoir1spp reservoir = Reservoir1spp::create();
    /*{
        float3 expected_irradiance = reprojected_gi_tex.SampleLevel(sampler_lnc, uv, 0).rgb;
        float p_q = max(1e-2, calculate_luma(expected_irradiance));
        reservoir.w_sum = p_q;
        reservoir.w_sel = p_q;
        reservoir.M = 1;
        reservoir.W = 1;
        outgoing_dir = mul(tangent_to_world, brdf_sample.wi);
        p_q_sel = p_q;
    }*/

    const uint reservoir_payload = px.x | (px.y << 16);

    reservoir.payload = reservoir_payload;

    if (brdf_sample.wi.z > 1e-5) {
        outgoing_dir = mul(tangent_to_world, brdf_sample.wi);

        RayDesc outgoing_ray;
        outgoing_ray.Direction = outgoing_dir;
        outgoing_ray.Origin = refl_ray_origin;
        outgoing_ray.TMin = 0;

        TraceResult result = do_the_thing(px, rng, outgoing_ray, gbuffer);
        const float p_q = calculate_luma(result.out_value) * brdf_sample.wi.z;
        float w_sum = p_q_sel = p_q;

        if (w_sum > 0) {
            reservoir.w_sum = w_sum;
            reservoir.payload = reservoir_payload;
            reservoir.W = 1;
            reservoir.M = 1;
        }
    }

    const float4 reproj = reprojection_tex[hi_px];

    const bool use_resampling = DIFFUSE_GI_USE_RESTIR;

    if (use_resampling) {
        float M_sum = reservoir.M;

        int2 reproj_px = (gbuffer_tex_size.xy * reproj.xy) / 2;

        for (uint sample_i = 0; sample_i < 1; ++sample_i) {
            int2 spx = px + reproj_px;

            float4 sample_gbuffer_packed = gbuffer_tex[spx * 2];
            GbufferData sample_gbuffer = GbufferDataPacked::from_uint4(asuint(sample_gbuffer_packed)).unpack();

            if (dot(sample_gbuffer.normal, gbuffer.normal) < 0.9) {
                continue;
            }

            const float4 prev_hit_ws_and_dist = ray_history_tex[spx];
            const float3 prev_hit_ws = prev_hit_ws_and_dist.xyz;
            const float prev_dist = prev_hit_ws_and_dist.w;

            if (!(prev_dist > 1e-8)) {
                continue;
            }

            const float3 prev_dir_unnorm = prev_hit_ws - refl_ray_origin;
            const float prev_dist_now = length(prev_dir_unnorm);
            const float3 prev_dir = normalize(prev_dir_unnorm);

            if (dot(prev_dir, gbuffer.normal) < 1e-3) {
                continue;
            }

            const float4 prev_irrad = irradiance_history_tex[spx];
            Reservoir1spp r = Reservoir1spp::from_raw(reservoir_history_tex[spx]);

            // From the ReSTIR paper:
            // With temporal reuse, the number of candidates M contributing to the
            // pixel can in theory grow unbounded, as each frame always combines
            // its reservoir with the previous frame’s. This causes (potentially stale)
            // temporal samples to be weighted disproportionately high during
            // resampling. To fix this, we simply clamp the previous frame’s M
            // to at most 20× of the current frame’s reservoir’s M
            r.M = min(r.M, 20);
            //r.M = min(r.M, 1);

            float p_q = 1;
            p_q *= calculate_luma(prev_irrad.rgb);
            p_q *= max(0, dot(prev_dir, gbuffer.normal));

            if (!(p_q > 0)) {
                continue;
            }

            float w = p_q * r.W * r.M;

            if (reservoir.update(w, reservoir_payload, rng)) {
                outgoing_dir = prev_dir;

                // mutate the direction
                // TODO: split off to a separate sample generation strategy
                #if 0
                    const float3 dir_ts = mul(outgoing_dir, tangent_to_world);
                    const float2 dir_pss = brdf.wi_to_primary_sample_space(dir_ts);

                    const float mut_x = uint_to_u01_float(hash1_mut(rng)) * 0.002;
                    const float mut_y = (uint_to_u01_float(hash1_mut(rng)) - 0.5) * 0.02;

                    float2 mut_dir_pss = frac(dir_pss + float2(mut_x, mut_y));
                    mut_dir_pss.y = abs(cos(acos(dir_pss.y) + mut_y));

                    outgoing_dir = mul(tangent_to_world, brdf.sample(wo, mut_dir_pss).wi);
                #endif

                p_q_sel = p_q;
            }

            M_sum += r.M;
        }

        reservoir.M = M_sum;
        reservoir.W = max(1e-5, reservoir.w_sum / (p_q_sel * reservoir.M));
    } else {
        outgoing_dir = mul(tangent_to_world, brdf_sample.wi);
    }

    RayDesc outgoing_ray;
    outgoing_ray.Direction = outgoing_dir;
    outgoing_ray.Origin = refl_ray_origin;
    outgoing_ray.TMin = 0;

    TraceResult result = do_the_thing(px, rng, outgoing_ray, gbuffer);

#if 1
    /*if (!use_resampling) {
        reservoir.w_sum = (calculate_luma(result.out_value));
        reservoir.w_sel = reservoir.w_sum;
        reservoir.W = 1;
        reservoir.M = 1;
    }*/

    irradiance_out_tex[px] = float4(result.out_value, dot(gbuffer.normal, outgoing_ray.Direction));
    hit0_out_tex[px] = float4(result.out_value * reservoir.W, 0);
    //hit0_out_tex[px] = float4(result.out_value, 0);
    ray_out_tex[px] = float4(outgoing_ray.Origin + outgoing_ray.Direction * result.hit_t, result.hit_t);
    reservoir_out_tex[px] = reservoir.as_raw();
#endif
}
