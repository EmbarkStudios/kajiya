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
#include "../inc/reservoir.hlsl"
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
[[vk::binding(10)]] Texture2D<float4> restir_irradiance_tex;
[[vk::binding(11)]] Texture2D<float4> restir_ray_tex;
[[vk::binding(12)]] Texture2D<float4> restir_reservoir_tex;
[[vk::binding(13)]] RWTexture2D<float4> output_tex;
[[vk::binding(14)]] RWTexture2D<float2> ray_len_output_tex;
[[vk::binding(15)]] cbuffer _ {
    float4 output_tex_size;
    int4 spatial_resolve_offsets[16 * 4 * 8];
};

#define SHUFFLE_SUBPIXELS 1
#define BORROW_SAMPLES 1
#define SHORT_CIRCUIT_NO_BORROW_SAMPLES 0

#define USE_APPROXIMATE_SAMPLE_SHADOWING 1

// Calculates hit points of neighboring samples, and rejects them if those land
// in the negative hemisphere of the sample being calculated.
// Adds quite a bit of ALU, but fixes some halos around corners.
#define REJECT_NEGATIVE_HEMISPHERE_REUSE 1

static const bool USE_RESTIR = true;
static const uint MAX_SAMPLE_COUNT = 8;

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

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
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

static float ggx_ndf_unnorm(float a2, float cos_theta) {
	float denom_sqrt = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
	return a2 / (denom_sqrt * denom_sqrt);
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
    
    const float a2 = max(RTR_ROUGHNESS_CLAMP, gbuffer.roughness) * max(RTR_ROUGHNESS_CLAMP, gbuffer.roughness);

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

    const float RAY_SQUISH_STRENGTH = 4;
    const float ray_len_avg = unsquish_ray_len(lerp(
        squish_ray_len(ray_len_history_tex.SampleLevel(sampler_lnc, uv + reprojection_params.xy, 0).y, RAY_SQUISH_STRENGTH),
        squish_ray_len(surf_to_hit_dist, RAY_SQUISH_STRENGTH),
        0.1), RAY_SQUISH_STRENGTH);

    const uint sample_count = BORROW_SAMPLES ? MAX_SAMPLE_COUNT : 1;

    float4 contrib_accum = 0.0;
    float ray_len_accum = 0;

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
    const float tan_theta = sqrt(gbuffer.roughness) * 0.25;
    
    const float kernel_size_vs = ray_len_avg / (ray_len_avg + eye_to_surf_dist);
    float kernel_size_ws = (kernel_size_vs * eye_to_surf_dist);
    kernel_size_ws *= tan_theta;

    // Clamp the kernel size so we don't sample the same point, but also don't thrash all the caches.
    // TODO: use pixels maybe
    kernel_size_ws = clamp(kernel_size_ws / eye_to_surf_dist, 0.0025, 0.025) * eye_to_surf_dist;
    
    float3 kernel_t1, kernel_t2;
    get_specular_filter_kernel_basis(
        -normalize(view_ray_context.ray_dir_ws()),
        gbuffer.normal,
        gbuffer.roughness,
        kernel_size_ws,
        kernel_t1, kernel_t2);

    uint rng = hash3(uint3(px, frame_constants.frame_index));

    // Offset to avoid correlation with ray generation
    float4 blue = blue_noise_for_pixel(half_px + 16, frame_constants.frame_index);

    // Feeds into the `pow` to remap sample index to radius.
    // At 0.5 (sqrt), it's proper circle sampling, with higher values becoming conical.
    // Must be constants, so the `pow` can be const-folded.
    const float KERNEL_SHARPNESS = 0.666;
    const float RADIUS_SAMPLE_MULT = 1.0 / pow(float(MAX_SAMPLE_COUNT - 1), KERNEL_SHARPNESS);

    // Bias away from the center sample to avoid half-res artifacts
    const float KERNEL_SIZE_BIAS = 0.16;

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        int2 sample_offset; {
            float ang = (sample_i + blue.x) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
            const float radius = pow(float(sample_i), KERNEL_SHARPNESS) * RADIUS_SAMPLE_MULT + KERNEL_SIZE_BIAS;

            float3 offset_ws = (cos(ang) * kernel_t1 + sin(ang) * kernel_t2) * radius;
            float3 sample_ws = view_ray_context.ray_hit_ws() + offset_ws;
            float3 sample_cs = position_world_to_clip(sample_ws);
            float2 sample_uv = cs_to_uv(sample_cs.xy);

            // Ad-hoc elongation fix based on stochastic SSR from Frostbite
            sample_uv.x = (sample_uv.x - uv.x) * lerp(saturate(wo.z * 2), 1.0, sqrt(gbuffer.roughness)) + uv.x;

            // TODO: pass in `input_tex_size`
            int2 sample_px = sample_uv * output_tex_size.xy / 2;
            sample_offset = sample_px - half_px;
        }
        
        int2 sample_px = half_px + sample_offset;
        float sample_depth = half_depth_tex[sample_px];
        float4 packed0 = hit0_tex[sample_px];

        const bool is_valid_sample = USE_RESTIR || packed0.w > 0;

        if (is_valid_sample && sample_depth != 0) {
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

            //const float4 sample_gbuffer_packed = gbuffer_tex[sample_px * 2 + hi_px_subpixels[frame_constants.frame_index & 3]];
            //GbufferData sample_gbuffer = GbufferDataPacked::from_uint4(asuint(sample_gbuffer_packed)).unpack();

            // Note: Not accurately normalized
            const float3 sample_hit_normal_vs = hit2_tex[sample_px].xyz;

            float3 sample_radiance;
            float sample_ray_pdf = 1;
            float neighbor_sampling_pdf;
            float3 center_to_hit_vs;
            float sample_cos_theta;
            float3 sample_hit_vs;

            if (USE_RESTIR) {
                uint2 rpx = sample_px;
                Reservoir1spp r = Reservoir1spp::from_raw(restir_reservoir_tex[rpx]);
                const uint2 spx = reservoir_payload_to_px(r.payload);
                const float3 sample_hit_ws = restir_ray_tex[spx].xyz;
                const float3 sample_offset = sample_hit_ws - view_ray_context.ray_hit_ws();
                const float sample_dist = length(sample_offset);
                const float3 sample_dir = sample_offset / sample_dist;

                sample_radiance = restir_irradiance_tex[spx].rgb;
                sample_ray_pdf = restir_irradiance_tex[spx].a;
                neighbor_sampling_pdf = 1.0 / r.W;
                center_to_hit_vs = position_world_to_view(sample_hit_ws) - lerp(view_ray_context.ray_hit_vs(), sample_origin_vs, RTR_NEIGHBOR_RAY_ORIGIN_CENTER_BIAS);
                sample_hit_vs = center_to_hit_vs + view_ray_context.ray_hit_vs();
                sample_cos_theta = restir_ray_tex[spx].a;
            } else {
                const float4 packed1 = hit1_tex[sample_px];
                neighbor_sampling_pdf = packed1.w;
                sample_radiance = packed0.xyz;
                sample_cos_theta = 1;

                #if RTR_RAY_HIT_STORED_AS_POSITION
                    center_to_hit_vs = packed1.xyz - lerp(view_ray_context.ray_hit_vs(), sample_origin_vs, RTR_NEIGHBOR_RAY_ORIGIN_CENTER_BIAS);
                    sample_hit_vs = center_to_hit_vs + view_ray_context.ray_hit_vs();
                #else
                    center_to_hit_vs = packed1.xyz;

                    // TODO
                    sample_hit_vs = 10000000000000000000;
                #endif
            }

            float3 wi = normalize(mul(direction_view_to_world(center_to_hit_vs), tangent_to_world));
            if (wi.z < 1e-5) {
                continue;
            }

            const float3 sample_normal_vs = half_view_normal_tex[sample_px].xyz;

            if (dot(normal_vs, sample_normal_vs) <= 0) {
                continue;
            }
            
        #if 0
            float rejection_bias = 1;
        #else
            float rejection_bias = 1;
            rejection_bias *= 0.01 + 0.99 * saturate(inverse_lerp(0.6, 0.9, dot(normal_vs, sample_normal_vs)));
            // overly aggressive: rejection_bias *= 0.01 + 0.99 * saturate(ggx_ndf_unnorm(a2, dot(normal_vs, sample_normal_vs)));
            // TODO: reject if roughness is vastly different
            rejection_bias *= exp2(-30.0 * abs(depth / sample_depth - 1.0));
            rejection_bias *= dot(sample_hit_normal_vs, center_to_hit_vs) < 0;
        #endif

        float center_to_hit_dist2 = dot(center_to_hit_vs, center_to_hit_vs);

        #if RTR_RAY_HIT_STORED_AS_POSITION
            // Compensate for change in visibility term. Automatic if storing with the surface area metric.
            #if !RTR_PDF_STORED_WITH_SURFACE_AREA_METRIC
                const float3 sample_to_hit_vs = sample_hit_vs - sample_origin_vs;
                float sample_to_hit_dist2 = dot(sample_to_hit_vs, sample_to_hit_vs);

                if (dot(sample_normal_vs, sample_to_hit_vs) <= 0) {
                    continue;
                }

                float center_to_hit_vis = dot(sample_hit_normal_vs, -normalize(center_to_hit_vs));
                float sample_to_hit_vis = dot(sample_hit_normal_vs, -normalize(sample_to_hit_vs));
                if (center_to_hit_vis <= 1e-5 || sample_to_hit_vis <= 1e-5) {
                    continue;
                }

                #if !RTR_APPROX_MEASURE_CONVERSION
                    neighbor_sampling_pdf /= clamp(wi.z / dot(sample_normal_vs, normalize(sample_to_hit_vs)), 0.25, 4.0);
                #endif
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

            BrdfValue spec = specular_brdf.evaluate(wo, wi);

            // The FG weight is included in the radiance accumulator,
            // so should not be in the ratio estimator weight.
            float spec_weight = spec.pdf * step(0.0, wi.z);

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

            float contrib_wt = 0;
            if (!USE_RESTIR) {
                contrib_wt = rejection_bias * spec_weight / neighbor_sampling_pdf;
                contrib_accum += float4(sample_radiance * spec.value_over_pdf, 1) * contrib_wt;
            } else {
                const float sample_ray_ndf = SpecularBrdf::ggx_ndf(a2, sample_cos_theta);
                const float center_ndf = SpecularBrdf::ggx_ndf(a2, normalize(wo + wi).z);

                float bent_sample_pdf = spec.pdf * sample_ray_ndf / center_ndf;

                // Blent towards using the real spec pdf at high roughness. Otherwise the ratio
                // estimator combined with bent sample rays results in too much brightness in corners/cracks.
                // At low roughness this can result in noise in the FG estimate, causing darkening.
                bent_sample_pdf = lerp(
                    bent_sample_pdf,
                    spec.pdf,
                    smoothstep(0.25, 0.75, sqrt(gbuffer.roughness))
                );

                bent_sample_pdf = min(bent_sample_pdf, RTR_RESTIR_MAX_PDF_CLAMP);

                // Pseudo MIS. Weigh down samples which claim to be high pdf
                // if we could have sampled them with a lower pdf. This helps rough reflections
                // surrounded by almost-mirrors. The rays sampled from the mirrors will have
                // high pdf values, and skew the integration, creating halos around themselves
                // on the rough objects.
                // This is similar in formulation to actual MIS; where we're lying is in claiming
                // that the central pixel generates samples, while it does not.
                //
                // The error from this seems to be making detail in roughness map hotter,
                // which is not a terrible thing given that it's also usually dimmed down
                // by temporal filters.
                float mis_weight = max(1e-5, spec.pdf / (sample_ray_pdf + spec.pdf));

                contrib_wt = rejection_bias * mis_weight * spec_weight / bent_sample_pdf;
                contrib_accum += float4(
                    sample_radiance * bent_sample_pdf / neighbor_sampling_pdf * spec.value_over_pdf,
                    1
                ) * contrib_wt;
            }

            float luma = calculate_luma(sample_radiance);
            ex += luma * contrib_wt;
            ex2 += luma * luma * contrib_wt;

            // Aggressively bias towards closer hits
            ray_len_accum += squish_ray_len(surf_to_hit_dist, RAY_SQUISH_STRENGTH) * contrib_wt;
        }
    }

    // TODO: when the borrow sample count is low (or 1), normalizing here
    // introduces a heavy bias, as it disregards the PDFs with which
    // samples have been taken. It should probably be temporally accumulated instead.

    const float contrib_norm_factor = max(1e-12, contrib_accum.w);
    //const float contrib_norm_factor = sample_count;

    contrib_accum.rgb /= contrib_norm_factor;
    ex /= contrib_norm_factor;
    ex2 /= contrib_norm_factor;
    ray_len_accum /= contrib_norm_factor;

    #if !RTR_RENDER_SCALED_BY_FG
        SpecularBrdfEnergyPreservation brdf_lut = SpecularBrdfEnergyPreservation::from_brdf_ndotv(specular_brdf, wo.z);
        contrib_accum.rgb /= brdf_lut.preintegrated_reflection;
    #endif

    ray_len_accum = unsquish_ray_len(ray_len_accum, RAY_SQUISH_STRENGTH);
    
    float3 out_color = contrib_accum.rgb;
    float relative_error = sqrt(max(0.0, ex2 - ex * ex)) / max(1e-5, ex2);
    //float relative_error = max(0.0, ex2 - ex * ex);

    //out_color /= brdf_lut.preintegrated_reflection;

    //relative_error = relative_error * 0.5 + 0.5 * WaveActiveMax(relative_error);
    //relative_error = WaveActiveMax(relative_error);

    //out_color = sample_count / 16.0;
    //out_color = saturate(filter_idx / 7.0);
    //out_color = relative_error * 0.01;
    //out_color = abs(ex2 - ex*ex) / max(1e-5, ex);

    //out_color = half_view_normal_tex[half_px].xyz * 0.5 + 0.5;
    output_tex[px] = float4(out_color, ex2);
    ray_len_output_tex[px] = float2(ray_len_accum, ray_len_avg);
}
