#include "../inc/color.hlsl"
#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/blue_noise.hlsl"
#include "rtr_settings.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> hit0_tex;
[[vk::binding(3)]] Texture2D<float4> hit1_tex;
[[vk::binding(4)]] Texture2D<float4> hit2_tex;
[[vk::binding(5)]] Texture2D<float4> history_tex;
[[vk::binding(6)]] Texture2D<float4> reprojection_tex;
[[vk::binding(7)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(8)]] Texture2D<float> half_depth_tex;
[[vk::binding(9)]] Texture2D<float2> ray_len_history_tex;
[[vk::binding(10)]] RWTexture2D<float4> output_tex;
[[vk::binding(11)]] RWTexture2D<float2> ray_len_output_tex;
[[vk::binding(12)]] cbuffer _ {
    float4 output_tex_size;
    int4 spatial_resolve_offsets[16 * 4 * 8];
};

#define USE_APPROX_BRDF 0
#define SHUFFLE_SUBPIXELS 1
#define BORROW_SAMPLES 1
#define SHORT_CIRCUIT_NO_BORROW_SAMPLES 0

// If true: Accumulate the full reflection * BRDF term, including FG
// If false: Accumulate lighting only, without the BRDF term
// TODO; true seems to have a better match with no borrowing
// ... actually, false seems better with the latest code.
#define ACCUM_FG_IN_RATIO_ESTIMATOR 0

#define USE_APPROXIMATE_SAMPLE_SHADOWING 1

// Calculates hit points of neighboring samples, and rejects them if those land
// in the negative hemisphere of the sample being calculated.
// Adds quite a bit of ALU, but fixes some halos around corners.
#define REJECT_NEGATIVE_HEMISPHERE_REUSE 1

float inverse_lerp(float minv, float maxv, float v) {
    return (v - minv) / (maxv - minv);
}

float approx_fresnel(float3 wo, float3 wi) {
   float3 h_unnorm = wo + wi;
   return exp2(-1.0 -8.65617024533378 * dot(wi, h_unnorm));
}

bool is_wave_alive(uint mask, uint idx) {
    return (mask & (1u << idx)) != 0;
}

// Get tangent vectors for the basis for specular filtering.
// Based on "Fast Denoising with Self Stabilizing Recurrent Blurs" by Dmitry Zhdan
void get_specular_filter_kernel_basis(float3 v, float3 n, float roughness, float scale, out float3 t1, out float3 t2) {
    float3 dominant = specular_dominant_direction(n, v, roughness);
    float3 reflected = reflect(-dominant, n);

    t1 = normalize(cross(n, reflected)) * scale;
    t2 = cross(reflected, t1);
}

// Encode ray length in a space which heavily favors short ones.
// For temporal averaging of distance to ray hits.
float squish_ray_len(float len, float squish_strength) {
    return exp2(-clamp(squish_strength * len, 0, 100));
}

// Ditto, decode.
float unsquish_ray_len(float len, float squish_strength) {
    return max(0.0, -1.0 / squish_strength * log2(1e-30 + len));
}

[numthreads(8, 8, 1)]
void main(const uint2 px : SV_DispatchThreadID) {
    const uint2 half_px = px / 2;

    const float2 uv = get_uv(px, output_tex_size);
    const float depth = depth_tex[px];

    if (0.0 == depth) {
        output_tex[px] = 0.0.xxxx;
        return;
    }

    #if !BORROW_SAMPLES && SHORT_CIRCUIT_NO_BORROW_SAMPLES
        output_tex[px] = float4(hit0_tex[half_px].rgb, 0.0);
        ray_len_output_tex[px] = 1;
        return;
    #endif

    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    const float4 gbuffer_packed = gbuffer_tex[px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    
    // Clamp to fix moire on mirror-like surfaces
    gbuffer.roughness = max(gbuffer.roughness, RTR_ROUGHNESS_CLAMP);

    const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
    float3 wo = mul(-normalize(view_ray_context.ray_dir_ws()), tangent_to_world);

    // Hack for shading normals facing away from the outgoing ray's direction:
    // We flip the outgoing ray along the shading normal, so that the reflection's curvature
    // continues, albeit at a lower rate.
    if (wo.z < 0.0) {
        wo.z *= -0.25;
        wo = normalize(wo);
    }

    SpecularBrdf specular_brdf;
    {
        LayeredBrdf layered_brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);
        specular_brdf = layered_brdf.specular_brdf;
    }
    const float f0_grey = calculate_luma(specular_brdf.albedo);

    // Index used to calculate a sample set disjoint for all four pixels in the quad
    // Offsetting by frame index reduces small structured artifacts
    const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + (SHUFFLE_SUBPIXELS ? 1 : 0) * frame_constants.frame_index) & 3;
    
    const float a2 = gbuffer.roughness * gbuffer.roughness;

    // Project lobe footprint onto the reflector, and find the desired convolution size
    #if RTR_RAY_HIT_STORED_AS_POSITION
        const float surf_to_hit_dist = length(hit1_tex[half_px].xyz - view_ray_context.ray_hit_vs());
    #else
        const float surf_to_hit_dist = length(hit1_tex[half_px].xyz);
    #endif
    const float eye_to_surf_dist = length(view_ray_context.ray_hit_vs());
    const float filter_spread = surf_to_hit_dist / (surf_to_hit_dist + eye_to_surf_dist);

    // Reduce size estimate variance by shrinking it across neighbors
    float filter_size = filter_spread * gbuffer.roughness * 16;

    const float4 reprojection_params = reprojection_tex[px];

    float history_error;
    {
        float4 history = history_tex.SampleLevel(sampler_lnc, uv + reprojection_params.xy, 0).w;
        float ex = calculate_luma(history.xyz);
        float ex2 = history.w;
        history_error = abs(ex * ex - ex2) / max(1e-8, ex);
    }

    const float ray_squish_strength = 4;

    const float ray_len_avg = unsquish_ray_len(lerp(
        squish_ray_len(ray_len_history_tex.SampleLevel(sampler_lnc, uv + reprojection_params.xy, 0).y, ray_squish_strength),
        squish_ray_len(surf_to_hit_dist, ray_squish_strength),
        0.1),ray_squish_strength);
    //const float ray_len_avg = unsquish_ray_len(ray_len_avg_squished);

    //history_error = lerp(2.0, history_error, reprojection_params.z);

    /*const uint wave_mask = WaveActiveBallot(true).x;
    if (is_wave_alive(wave_mask, WaveGetLaneIndex() ^ 2)) {
        filter_size = min(filter_size, WaveReadLaneAt(filter_size, WaveGetLaneIndex() ^ 2));
    }
    if (is_wave_alive(wave_mask, WaveGetLaneIndex() ^ 16)) {
        filter_size = min(filter_size, WaveReadLaneAt(filter_size, WaveGetLaneIndex() ^ 16));
    }*/

    // Expand the filter size if variance is high, but cap it, so we don't destroy contact reflections
    const float error_adjusted_filter_size = min(filter_size * 4, filter_size + history_error * 0.5);

    //const uint sample_count = BORROW_SAMPLES ? clamp(history_error * 0.5 * 16 * saturate(filter_size * 8), 4, 16) : 1;
    //sample_count = WaveActiveMax(sample_count);
    const uint sample_count = BORROW_SAMPLES ? 8 : 1;
    //const uint sample_count = BORROW_SAMPLES ? clamp(error_adjusted_filter_size * 128, 6, 16) : 1;

    // Choose one of a few pre-baked sample sets based on the footprint
    const uint filter_idx = uint(clamp(error_adjusted_filter_size * 8, 0, 7));
    //const uint filter_idx = 3;
    //output_tex[px] = float4((filter_idx / 7.0).xxx, 0);
    //return;

    //SpecularBrdfEnergyPreservation brdf_lut = SpecularBrdfEnergyPreservation::from_brdf_ndotv(specular_brdf, wo.z);

    float4 contrib_accum = 0.0;
    float ray_len_accum = 0;
    float3 brdf_weight_accum = 0.0;

    float ex = 0.0;
    float ex2 = 0.0;

    const float3 normal_vs = direction_world_to_view(gbuffer.normal);

    // The footprint of our kernel increases with roughness, and scales with distance.
    // The latter is a simple relationship of distance to camera and distance to hit point,
    // and for the roughness factor we can bound a cone.
    //
    // According to 4-9-3-DistanceBasedRoughnessLobeBounding.nb
    // in "Moving Frostbite to PBR" by Seb and Charles, the cutoff angle tangent should be:
    // sqrt(energy_frac / (1.0 - energy_frac)) * roughness
    // That doesn't work however, resulting in a kernel which grows too slowly with
    // increasing roughness. Unclear why -- related to the cone being derived
    // for the NDF which exists in half-angle space?
    //
    // What does work however is a square root relationship (constants arbitrary):
    const float tan_theta = sqrt(gbuffer.roughness) * 0.1;
    
    const float kernel_size_vs = ray_len_avg / (ray_len_avg + eye_to_surf_dist);
    float kernel_size_ws = (kernel_size_vs * eye_to_surf_dist);
    //kernel_size_ws += 0.05 * history_error;
    kernel_size_ws *= tan_theta;

    // Clamp the kernel size so we don't sample the same point, but also don't thrash all the caches.
    // TODO: use pixels maybe
    kernel_size_ws = clamp(kernel_size_ws / eye_to_surf_dist, 0.001, 0.01) * eye_to_surf_dist;
    
    float3 kernel_t1, kernel_t2;
    get_specular_filter_kernel_basis(
        -normalize(view_ray_context.ray_dir_ws()),
        gbuffer.normal,
        gbuffer.roughness,
        kernel_size_ws,
        kernel_t1, kernel_t2);

    // Offset to avoid correlation with ray generation
    float4 blue = blue_noise_for_pixel(half_px + 16, frame_constants.frame_index);

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
    //for (uint sample_i = 7; sample_i < 8; ++sample_i) {
    //for (uint sample_i_ = 0; sample_i_ < 8; ++sample_i_) { uint sample_i = sample_i_ == 0 ? 0 : sample_i_ + 8;

        #if 0
            // TODO: precalculate temporal variants
            const int2 sample_offset = spatial_resolve_offsets[(px_idx_in_quad * 16 + sample_i) + 64 * filter_idx].xy;
        #elif 0
            int2 sample_offset; {
                float ang = (sample_i + blue.x) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
                float radius_inc = lerp(0.333, 1.0, saturate(error_adjusted_filter_size / 8));
                float radius = 1.5 + float(sample_i) * radius_inc;
                sample_offset = float2(cos(ang), sin(ang)) * radius;
            }
        #else
            int2 sample_offset; {
                float ang = (sample_i + blue.x) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
                //float radius_inc = lerp(0.333, 1.0, saturate(error_adjusted_filter_size / 8));
                float radius = (float(sample_i) + 0.5) * 0.5;

                float3 offset_ws = (cos(ang) * kernel_t1 + sin(ang) * kernel_t2) * radius;
                float3 sample_ws = view_ray_context.ray_hit_ws() + offset_ws;
                float3 sample_cs = position_world_to_clip(sample_ws);
                float2 sample_uv = cs_to_uv(sample_cs.xy);
                // ad-hoc elongation fix based on stochastic ssr slides
                //sample_uv = (sample_uv - uv) * float2(lerp(saturate(wo.z * 2), 1.0, sqrt(gbuffer.roughness)), 1.0) + uv;
                int2 sample_px = sample_uv * output_tex_size.xy / 2;
                sample_offset = sample_px - half_px;
            }
        #endif
        
        int2 sample_px = half_px + sample_offset;
        float sample_depth = half_depth_tex[sample_px];

        float4 packed0 = hit0_tex[sample_px];

        if (packed0.w > 0 && sample_depth != 0) {
            // Note: must match the raygen
            uint2 hi_px_subpixels[4] = {
                uint2(0, 0),
                uint2(1, 1),
                uint2(1, 0),
                uint2(0, 1),
            };
            const float2 sample_uv = get_uv(
                sample_px * 2 + hi_px_subpixels[frame_constants.frame_index & 3],
                output_tex_size);

            const ViewRayContext sample_ray_ctx = ViewRayContext::from_uv_and_depth(sample_uv, sample_depth);
            const float3 sample_origin_vs = sample_ray_ctx.ray_hit_vs();

            float4 packed1 = hit1_tex[sample_px];
            float neighbor_sampling_pdf = packed1.w;

            // Note: Not accurately normalized
            const float3 sample_hit_normal_vs = hit2_tex[sample_px].xyz;

            #if RTR_RAY_HIT_STORED_AS_POSITION
                const float3 center_to_hit_vs = packed1.xyz - lerp(view_ray_context.ray_hit_vs(), sample_origin_vs, RTR_NEIGHBOR_RAY_ORIGIN_CENTER_BIAS);
                const float3 sample_hit_vs = center_to_hit_vs + view_ray_context.ray_hit_vs();
            #else
                const float3 center_to_hit_vs = packed1.xyz;
            #endif

            {
                //output_tex[px] = float4(normalize(direction_view_to_world(sample_hit_normal_vs)) * 0.5 + 0.5, 1);
                //output_tex[px] = float4(normalize(direction_view_to_world(center_to_hit_vs)) * 0.5 + 0.5, 1);
                //return;
            }

            float3 wi = normalize(mul(direction_view_to_world(center_to_hit_vs), tangent_to_world));

            float sample_hit_to_center_vis = 1;

            if (wi.z > 1e-5) {
            #if 0
                float rejection_bias = 1;
            #else
                float3 sample_normal_vs = half_view_normal_tex[sample_px].xyz;
                // TODO: brdf-driven blend
                float rejection_bias =
                    0.01 + 0.99 * saturate(inverse_lerp(0.6, 0.9, dot(normal_vs, sample_normal_vs)));
                //rejection_bias *= exp2(-10.0 * abs(1.0 / sample_depth - 1.0 / depth) * depth);
                rejection_bias *= exp2(-10.0 * abs(depth / sample_depth - 1.0));
                rejection_bias *= dot(sample_hit_normal_vs, center_to_hit_vs) < 0;

                sample_hit_to_center_vis = saturate(dot(sample_hit_normal_vs, -normalize(center_to_hit_vs)));
            #endif

            float center_to_hit_dist2 = dot(center_to_hit_vs, center_to_hit_vs);

            #if RTR_RAY_HIT_STORED_AS_POSITION
                // Compensate for change in visibility term. Automatic if storing with the surface area metric.
                #if !RTR_PDF_STORED_WITH_SURFACE_AREA_METRIC
                    const float3 sample_to_hit_vs = sample_hit_vs - sample_origin_vs;
                    float sample_to_hit_dist2 = dot(sample_to_hit_vs, sample_to_hit_vs);

                    neighbor_sampling_pdf /=
                        (1.0 / max(1e-20, center_to_hit_dist2))
                        / (1.0 / max(1e-20, sample_to_hit_dist2));

                        float center_to_hit_vis = saturate(dot(sample_hit_normal_vs, -normalize(center_to_hit_vs)));
                        float sample_to_hit_vis = saturate(dot(sample_hit_normal_vs, -normalize(sample_to_hit_vs)));
                        if (center_to_hit_vis <= 1e-5 || sample_to_hit_vis <= 1e-5) {
                            continue;
                        }

                        neighbor_sampling_pdf /= center_to_hit_vis / sample_to_hit_vis;
                        //neighbor_sampling_pdf /=  max(1e-5, dot(normal_vs, normalize(center_to_hit_vs))) / max(1e-5, dot(sample_normal_vs, normalize(sample_to_hit_vs)));
                        neighbor_sampling_pdf /= max(1e-5, wi.z) / max(1e-5, dot(sample_normal_vs, normalize(sample_to_hit_vs)));
                #endif

        		float3 surface_offset = sample_origin_vs - view_ray_context.ray_hit_vs();
                #if USE_APPROXIMATE_SAMPLE_SHADOWING
            		if (dot(center_to_hit_vs, normal_vs) * 0.2 / length(center_to_hit_vs) < dot(surface_offset, normal_vs) / length(surface_offset)) {
            			rejection_bias *= sample_i == 0 ? 1 : 0;
            		}
                #endif
            #else
                #if REJECT_NEGATIVE_HEMISPHERE_REUSE
                    const float3 sample_hit_vs = sample_origin_vs + center_to_hit_vs;
                    rejection_bias *= dot(sample_hit_vs - view_ray_context.ray_hit_vs(), normal_vs) > 0.0;
                #endif
            #endif

    #if !USE_APPROX_BRDF
                BrdfValue spec = specular_brdf.evaluate(wo, wi);

                // Note: looks closer to reference when comparing against metalness=1 albedo=1 PT reference.
                // As soon as the regular image is compared though. This term just happens to look
                // similar to the lack of Fresnel in the metalness=1 albedo=1 case.
                // float spec_weight = spec.value.x * max(0.0, wi.z);

                #if ACCUM_FG_IN_RATIO_ESTIMATOR
                    // Note: could be spec.pdf too, though then fresnel isn't accounted for in the weights
                    float spec_weight = calculate_luma(spec.value) * step(0.0, wi.z);
                #else
                    float spec_weight = calculate_luma(spec.value) * step(0.0, wi.z);
                #endif
    #else
                float spec_weight;
                //{
                    const float3 m = normalize(wo + wi);
                    const float cos_theta = m.z;
                    const float pdf_h_over_cos_theta = SpecularBrdf::ggx_ndf(a2, cos_theta);
                    //const float f = lerp(f0_grey, 1.0, approx_fresnel(wo, wi));
                    const float3 f = eval_fresnel_schlick(specular_brdf.albedo, 1.0, dot(m, wi)).x;

                    spec_weight = calculate_luma(f) * pdf_h_over_cos_theta * step(0.0, wi.z) / max(1e-5, wo.z + wi.z);
                //}
    #endif

                float to_psa_metric = 1;
                #if RTR_PDF_STORED_WITH_SURFACE_AREA_METRIC
                    // Convert to projected solid angle
                    to_psa_metric =
                        #if RTR_APPROX_MEASURE_CONVERSION
                            1
                        #else
                            abs(wi.z * dot(sample_hit_normal_vs, -normalize(center_to_hit_vs)))
                        #endif
                        / center_to_hit_dist2;

                    neighbor_sampling_pdf /= to_psa_metric;
                #endif

                float mis_weight = 1;

                // TODO: this is all a lie, and pretends that we always have the central sample to use.
                // especially with reservoir exchange, sample0 might not even be close to the central sample.
                //
                // The aim here is to address the issue of a low-prob sample landing in a high-prob area
                // of the center pixel, and blowing it up.
                //
                // TODO: actually seems to distort the lobe, so maybe nuke it.
                //if (sample_i > 0)
                {
                    // TODO: this borks things when using reservoir exchange because
                    // the PDFs in neighbor_sampling_pdf are reservoir PDFs and not BRDF PDFs
                    //mis_weight = neighbor_sampling_pdf / (spec.pdf + neighbor_sampling_pdf);

                    // YOLO fix; only valid when using exhange
                    //mis_weight = packed0.w / (spec.pdf * to_psa_metric + packed0.w);
                }

                //spec_weight *= sample_hit_to_center_vis;

                #if !ACCUM_FG_IN_RATIO_ESTIMATOR
                    float contrib_wt = rejection_bias * step(0.0, wi.z) * spec_weight / neighbor_sampling_pdf * mis_weight;
                    //contrib_wt = 1;
                    contrib_accum += float4(
                        packed0.rgb
                        ,
                        1
                    ) * contrib_wt;
                #else
                    float contrib_wt = rejection_bias * step(0.0, wi.z) * spec_weight / neighbor_sampling_pdf * mis_weight;
                    contrib_accum += float4(
                        packed0.rgb * spec.value_over_pdf
                        ,
                        1
                    ) * contrib_wt;
                #endif

                brdf_weight_accum += spec.value_over_pdf * contrib_wt;

                float luma = calculate_luma(packed0.rgb);
                ex += luma * contrib_wt;
                ex2 += luma * luma * contrib_wt;

                // Aggressively bias towards closer hits
                ray_len_accum += squish_ray_len(surf_to_hit_dist, ray_squish_strength) * contrib_wt;
            }
        }
    }

    // TODO: when the borrow sample count is low (or 1), normalizing here
    // introduces a heavy bias, as it disregards the PDFs with which
    // samples have been taken. It should probably be temporally accumulated instead.
    //
    // Could temporally accumulate contributions scaled by brdf_weight_accum,
    // but then the latter needs to be stored as well, and renormalized before
    // sampling in a given frame. Could also be tricky to temporally filter it.

    const float contrib_norm_factor = max(1e-8, contrib_accum.w);
    //const float contrib_norm_factor = sample_count;

    contrib_accum.rgb /= contrib_norm_factor;
    ex /= contrib_norm_factor;
    ex2 /= contrib_norm_factor;
    ray_len_accum /= contrib_norm_factor;

    brdf_weight_accum /= max(1e-8, contrib_accum.w);

    #if !RTR_RENDER_SCALED_BY_FG && ACCUM_FG_IN_RATIO_ESTIMATOR
        // Un-scale by the FG term so we can denoise just the lighting,
        // and the darkening effect of FG becomes temporally responsive.
        contrib_accum.rgb /= max(1e-8, brdf_weight_accum);
    #elif RTR_RENDER_SCALED_BY_FG && !ACCUM_FG_IN_RATIO_ESTIMATOR
        // Edge case for debug only
        contrib_accum.rgb *= max(1e-8, brdf_weight_accum);
    #endif

    ray_len_accum = unsquish_ray_len(ray_len_accum, ray_squish_strength);
    
    float3 out_color = contrib_accum.rgb;
    float relative_error = sqrt(max(0.0, ex2 - ex * ex)) / max(1e-5, ex2);
    //float relative_error = max(0.0, ex2 - ex * ex);

    //out_color /= brdf_lut.preintegrated_reflection;

    //relative_error = relative_error * 0.5 + 0.5 * WaveActiveMax(relative_error);
    //relative_error = WaveActiveMax(relative_error);

    //out_color = sample_count / 16.0;
    //out_color = saturate(filter_idx / 7.0);
    //out_color = relative_error;
    //out_color = abs(ex2 - ex*ex) / max(1e-5, ex);
    //out_color = history.w;

    //out_color = half_view_normal_tex[half_px].xyz * 0.5 + 0.5;
    output_tex[px] = float4(out_color, ex2);
    ray_len_output_tex[px] = float2(ray_len_accum, ray_len_avg);
}
