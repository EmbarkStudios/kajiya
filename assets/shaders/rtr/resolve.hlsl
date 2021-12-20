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

#define SHUFFLE_SUBPIXELS 1
#define BORROW_SAMPLES 1
#define SHORT_CIRCUIT_NO_BORROW_SAMPLES 0

#define USE_APPROXIMATE_SAMPLE_SHADOWING 1

// Note: Only does anything if RTR_RAY_HIT_STORED_AS_POSITION is 0
// Calculates hit points of neighboring samples, and rejects them if those land
// in the negative hemisphere of the sample being calculated.
// Adds quite a bit of ALU, but fixes some halos around corners.
#define REJECT_NEGATIVE_HEMISPHERE_REUSE 1

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

// Get tangent vectors for the basis for specular filtering.
// Based on "Fast Denoising with Self Stabilizing Recurrent Blurs" by Dmitry Zhdan
void get_specular_filter_kernel_basis(float3 v, float3 n, float roughness, float scale, out float3 t1, out float3 t2) {
    float3 dominant = specular_dominant_direction(n, v, roughness);
    float3 reflected = reflect(-dominant, n);

    t1 = normalize(cross(n, reflected)) * scale;
    t2 = cross(reflected, t1);
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

    // Needed to account for perspective distortion to keep the kernel constant
    // near screen boundaries at high FOV.
    const float eye_ray_z_scale = -view_ray_context.ray_dir_vs().z;

    const float4 reprojection_params = reprojection_tex[px];

    const float RAY_SQUISH_STRENGTH = 4;
    const float ray_len_avg = exponential_unsquish(lerp(
        exponential_squish(ray_len_history_tex.SampleLevel(sampler_lnc, uv + reprojection_params.xy, 0).y, RAY_SQUISH_STRENGTH),
        exponential_squish(surf_to_hit_dist, RAY_SQUISH_STRENGTH),
        0.1),RAY_SQUISH_STRENGTH);

    const uint sample_count = BORROW_SAMPLES ? MAX_SAMPLE_COUNT : 1;

    float4 contrib_accum = 0.0;
    float ray_len_accum = 0;

    float ex = 0.0;
    float ex2 = 0.0;

    const float3 normal_vs = direction_world_to_view(gbuffer.normal);

    static const float GOLDEN_ANGLE = 2.39996323;

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
    
    float kernel_size_ws;
    {
        // Clamp the ray length used in kernel size calculations, so we don't end up with
        // tiny kernels in corners. In the presense of fireflies (e.g. from reflected spec),
        // that would result in small circles appearing as reflections.
        const float clamped_ray_len_avg = max(
            ray_len_avg,
            eye_to_surf_dist / eye_ray_z_scale * frame_constants.view_constants.clip_to_view[1][1] * 0.2 * 1
        );

        const float kernel_size_vs = clamped_ray_len_avg / (clamped_ray_len_avg + eye_to_surf_dist);
        kernel_size_ws = kernel_size_vs * eye_to_surf_dist * eye_ray_z_scale;
        kernel_size_ws *= tan_theta;
    }

    {
        const float scale_factor = eye_to_surf_dist * eye_ray_z_scale * frame_constants.view_constants.clip_to_view[1][1];

        // Clamp the kernel size so we don't sample the same point, but also don't thrash all the caches.
        kernel_size_ws = clamp(kernel_size_ws, output_tex_size.w * 2.0 * scale_factor, 0.1 * scale_factor);
    }
    
    float3 kernel_t1, kernel_t2;
    get_specular_filter_kernel_basis(
        -normalize(view_ray_context.ray_dir_ws()),
        gbuffer.normal,
        gbuffer.roughness,
        kernel_size_ws,
        kernel_t1, kernel_t2);

    // Offset to avoid correlation with ray generation
    float4 blue = blue_noise_for_pixel(half_px + 16, frame_constants.frame_index);

    // Feeds into the `pow` to remap sample index to radius.
    // At 0.5 (sqrt), it's disk sampling, with higher values becoming conical.
    // Must be constant, so the `pow` can be const-folded.
    const float KERNEL_SHARPNESS = 0.666;
    //const float KERNEL_SHARPNESS = 0.5;
    const float RADIUS_SAMPLE_MULT = 1.0 / pow(float(MAX_SAMPLE_COUNT), KERNEL_SHARPNESS);

    //const float ang_offset = blue.x;
    // Way faster, seems to look the same.
    const float ang_offset = (frame_constants.frame_index * 59 % 128) * M_PLASTIC;

    // Go from the outside to the inside. This is important for sampling of the central contribution,
    // as we might need to disable jitter for it based on the suitability of the neighborhood.
    for (int sample_i = sample_count - 1; sample_i >= 0; --sample_i) {
        int2 sample_offset; {
            float ang = (sample_i + ang_offset) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;

            float sample_i_with_jitter = sample_i;

            // If we've arrived at the central sample, and haven't found satisfactory
            // contributions yet, load the candidate directly from `half_px`.
            // Otherwise jitter the location to avoid half-pixel artifacts.
            // This additional check reduces undersampling on thin shiny surfaces,
            // which otherwise causes them to appear black.
            if (sample_i == 0) {
                const bool offset_center_pixel = true
                    // roughness check is a HACK. it's only there because not offsetting
                    // the center pixel on rough surfaces causes pixellation.
                    // it can happen for thin features, or features seen at an angle.
                    && (contrib_accum.w > 1e-2 || gbuffer.roughness > 0.1)
                    && reprojection_params.z == 1.0
                    ;

                if (offset_center_pixel) {
                    sample_i_with_jitter += blue.y;
                }
            } else {
                sample_i_with_jitter += blue.y;
            }

            const float radius = BORROW_SAMPLES
                ? pow(sample_i_with_jitter, KERNEL_SHARPNESS) * RADIUS_SAMPLE_MULT
                : 0;

            float3 offset_ws = (cos(ang) * kernel_t1 + sin(ang) * kernel_t2) * radius;
            float3 sample_ws = view_ray_context.ray_hit_ws() + offset_ws;
            float3 sample_cs = position_world_to_sample(sample_ws);
            float2 sample_uv = cs_to_uv(sample_cs.xy);

            // TODO: pass in `input_tex_size`
            int2 sample_px = int2(floor(sample_uv * output_tex_size.xy / 2));
            sample_offset = sample_px - half_px;
        }

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

            // Discard hit samples which face away from the center pixel
            if (dot(sample_hit_normal_vs, -center_to_hit_vs) <= 0) {
                continue;
            }

            float3 wi = normalize(mul(direction_view_to_world(center_to_hit_vs), tangent_to_world));

            float sample_hit_to_center_vis = 1;

            if (wi.z > 1e-5) {
                float3 sample_normal_vs = half_view_normal_tex[sample_px].xyz;

                // Soft directional falloff; TODO: brdf-driven blend
                float rejection_bias =
                    0.01 + 0.99 * saturate(inverse_lerp(0.6, 0.9, dot(normal_vs, sample_normal_vs)));
                // Depth
                rejection_bias *= exp2(-10.0 * abs(depth / sample_depth - 1.0));

                sample_hit_to_center_vis = saturate(dot(sample_hit_normal_vs, -normalize(center_to_hit_vs)));

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

                BrdfValue spec = specular_brdf.evaluate(wo, wi);

                // The FG weight is included in the radiance accumulator,
                // so should not be in the ratio estimator weight.
                const float spec_weight = spec.pdf * step(0.0, wi.z);

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

                const float contrib_wt = rejection_bias * step(0.0, wi.z) * spec_weight / neighbor_sampling_pdf;
                contrib_accum += float4(
                    packed0.rgb * spec.value_over_pdf
                    ,
                    1
                ) * contrib_wt;

                float luma = calculate_luma(packed0.rgb);
                ex += luma * contrib_wt;
                ex2 += luma * luma * contrib_wt;

                // Aggressively bias towards closer hits
                ray_len_accum += exponential_squish(surf_to_hit_dist, RAY_SQUISH_STRENGTH) * contrib_wt;
            }
        }
    }

    // Note: When the borrow sample count is low (or 1), normalizing here
    // introduces a heavy bias, as it disregards the PDFs with which
    // samples have been taken.

    const float contrib_norm_factor = max(1e-8, contrib_accum.w);

    contrib_accum.rgb /= contrib_norm_factor;
    ex /= contrib_norm_factor;
    ex2 /= contrib_norm_factor;
    ray_len_accum /= contrib_norm_factor;

    SpecularBrdfEnergyPreservation brdf_lut = SpecularBrdfEnergyPreservation::from_brdf_ndotv(specular_brdf, wo.z);

    #if !RTR_RENDER_SCALED_BY_FG
        contrib_accum.rgb /= brdf_lut.preintegrated_reflection;
    #endif

    //contrib_accum.rgb *= brdf_lut.preintegrated_reflection_mult;

    ray_len_accum = exponential_unsquish(ray_len_accum, RAY_SQUISH_STRENGTH);
    
    float3 out_color = contrib_accum.rgb;
    float relative_error = sqrt(max(0.0, ex2 - ex * ex)) / max(1e-5, ex2);

    output_tex[px] = float4(out_color, ex2);
    ray_len_output_tex[px] = float2(ray_len_accum, ray_len_avg);
}
