#include "../inc/color.hlsl"
#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
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
[[vk::binding(9)]] RWTexture2D<float4> hit0_output_tex;
[[vk::binding(10)]] RWTexture2D<float4> hit1_output_tex;
[[vk::binding(11)]] RWTexture2D<float4> hit2_output_tex;
[[vk::binding(12)]] cbuffer _ {
    float4 output_tex_size;
    int4 spatial_resolve_offsets[16 * 4 * 8];
};

#define USE_APPROX_BRDF 0
#define BORROW_SAMPLES 1
#define USE_APPROXIMATE_SAMPLE_SHADOWING 1
#define USE_VISIBILITY_TERM_COMPENSATION 0

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

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    // TODO: must match the UV in raygen
    //const float2 uv = get_uv(px, output_tex_size);
    const float2 uv = get_uv(px * 2, output_tex_size * float4(2.0.xx, 0.5.xx));

    const float depth = half_depth_tex[px];

    if (0.0 == depth) {
        return;
    }

    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    const float4 gbuffer_packed = gbuffer_tex[px * 2];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    
    if (!BORROW_SAMPLES/* || gbuffer.roughness < 0.1*/) {
        hit0_output_tex[px] = hit0_tex[px];
        hit1_output_tex[px] = hit1_tex[px];
        hit2_output_tex[px] = hit2_tex[px];
        return;
    }

    // Clamp to fix moire on mirror-like surfaces
    gbuffer.roughness = max(gbuffer.roughness, 3e-4);

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
    const float a2 = gbuffer.roughness * gbuffer.roughness;

    // Project lobe footprint onto the reflector, and find the desired convolution size
    #if RTR_RAY_HIT_STORED_AS_POSITION
        const float surf_to_hit_dist = length(hit1_tex[px].xyz - view_ray_context.ray_hit_vs());
    #else
        const float surf_to_hit_dist = length(hit1_tex[px].xyz);
    #endif
    const float eye_to_surf_dist = -view_ray_context.ray_hit_vs().z;
    const float filter_spread = surf_to_hit_dist / (surf_to_hit_dist + eye_to_surf_dist);

    // Reduce size estimate variance by shrinking it across neighbors
    float filter_size = filter_spread * gbuffer.roughness * 16;

    /*const float4 reprojection_params = reprojection_tex[px];

    float history_error;
    {
        float4 history = history_tex.SampleLevel(sampler_lnc, uv + reprojection_params.xy, 0).w;
        float ex = calculate_luma(history.xyz);
        float ex2 = history.w;
        history_error = abs(ex * ex - ex2) / max(1e-8, ex);
    }
    //history_error = lerp(2.0, history_error, reprojection_params.z);

    const uint wave_mask = WaveActiveBallot(true).x;
    if (is_wave_alive(wave_mask, WaveGetLaneIndex() ^ 2)) {
        filter_size = min(filter_size, WaveReadLaneAt(filter_size, WaveGetLaneIndex() ^ 2));
    }
    if (is_wave_alive(wave_mask, WaveGetLaneIndex() ^ 16)) {
        filter_size = min(filter_size, WaveReadLaneAt(filter_size, WaveGetLaneIndex() ^ 16));
    }*/

    // Expand the filter size if variance is high, but cap it, so we don't destroy contact reflections
    //const float error_adjusted_filter_size = min(filter_size * 4, filter_size + history_error * 0.5);
    const float error_adjusted_filter_size = filter_size;

    //const uint sample_count = BORROW_SAMPLES ? clamp(history_error * 0.5 * 16 * saturate(filter_size * 8), 4, 16) : 1;
    //sample_count = WaveActiveMax(sample_count);
    uint sample_count = BORROW_SAMPLES ? 16 : 1;
    //const uint sample_count = BORROW_SAMPLES ? clamp(error_adjusted_filter_size * 128, 6, 16) : 1;

    // HACK workaround for excessive blurring near contacts
    if (filter_spread < 0.3) {
        //sample_count = 1;
        hit0_output_tex[px] = hit0_tex[px];
        hit1_output_tex[px] = hit1_tex[px];
        hit2_output_tex[px] = hit2_tex[px];
        return;
    }

    // Choose one of a few pre-baked sample sets based on the footprint
    const uint filter_idx = uint(clamp(error_adjusted_filter_size * 8, 0, 7));

    float4 reservoir_value0 = 0.0.xxxx;
    float4 reservoir_value1 = 0.0.xxxx;
    float4 reservoir_value2 = 0.0.xxxx;
    float reservoir_rate = 0.0;
    float reservoir_pdf = 0.0;
    float reservoir_pdf_wt_sum = 0.0;
    float reservoir_rate_sum = 0.0;
    float reservoir_rate_weighted_sum = 0.0;
    float reservoir_pdf_sum = 0.0;
    float valid_sample_count = 0.0;
    float picked_brdf_val = 0.0;
    float brdf_val_sum = 0.0;

    uint rng = hash3(uint3(px, frame_constants.frame_index));
    //uint rng = hash2(px);

    const float3 normal_vs = mul(frame_constants.view_constants.world_to_view, float4(gbuffer.normal, 0)).xyz;

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        // TODO: precalculate temporal variants
        //const int2 sample_offset = spatial_resolve_offsets[sample_i + (rng % 4 * 16) + 64 * filter_idx].xy;
        const int2 sample_offset = 2 * int2(1, 1) * spatial_resolve_offsets[sample_i + (rng % 4 * 16) + 64 * filter_idx].xy;
        const int2 sample_px = px + sample_offset;
        float sample_depth = half_depth_tex[sample_px];

        float4 packed0 = hit0_tex[sample_px];
        if (packed0.w > 0 && sample_depth != 0) {
            const float2 sample_uv = get_uv(sample_px, output_tex_size);
            const ViewRayContext sample_ray_ctx = ViewRayContext::from_uv_and_depth(sample_uv, sample_depth);
            const float3 sample_origin_vs = sample_ray_ctx.ray_hit_vs();

            float4 packed1 = hit1_tex[sample_px];
            float neighbor_sampling_pdf = packed1.w;

            float4 packed2 = hit2_tex[sample_px];
            // Note: Not accurately normalized
            const float3 sample_hit_normal_vs = packed2.xyz;

            #if RTR_RAY_HIT_STORED_AS_POSITION
                const float3 sample_hit_vs = packed1.xyz;
                const float3 center_to_hit_vs = sample_hit_vs - lerp(view_ray_context.ray_hit_vs(), sample_origin_vs, RTR_NEIGHBOR_RAY_ORIGIN_CENTER_BIAS);
            #else
                const float3 center_to_hit_vs = packed1.xyz;
                const float3 sample_hit_vs = packed1.xyz + view_ray_context.ray_hit_vs();
            #endif

            float3 wi = mul(normalize(direction_view_to_world(center_to_hit_vs)), tangent_to_world);

            //float3 wi = mul(normalize(packed1.xyz), tangent_to_world);

            if (wi.z > 1e-5) {

            #if 0
                float rejection_bias = 1;
            #else
                float3 sample_normal_vs = half_view_normal_tex[sample_px].xyz;
#if RTR_RAY_HIT_STORED_AS_POSITION
                float rejection_bias =
                    saturate(inverse_lerp(0.6, 0.9, dot(normal_vs, sample_normal_vs)));
#else
                float rejection_bias =
                    saturate(inverse_lerp(0.2, 0.6, dot(normal_vs, sample_normal_vs)));
#endif
                //rejection_bias *= exp2(-10.0 * abs(1.0 / sample_depth - 1.0 / depth) * depth);
                rejection_bias *= exp2(-10.0 * abs(depth / sample_depth - 1.0));

                // Reject neighbor hits which don't see the center pixel (negative hemisphere of hits)
                //rejection_bias *= dot(sample_hit_normal_vs, center_to_hit_vs) < 0;
            #endif
            
            const float3 sample_to_hit_vs = sample_hit_vs - sample_origin_vs;
            float sample_to_hit_dist2 = dot(sample_to_hit_vs, sample_to_hit_vs);
            float center_to_hit_dist2 = dot(center_to_hit_vs, center_to_hit_vs);

    		float3 surface_offset = sample_origin_vs - view_ray_context.ray_hit_vs();
            #if USE_APPROXIMATE_SAMPLE_SHADOWING
        		if (dot(center_to_hit_vs, normal_vs) * 0.2 / length(center_to_hit_vs) < dot(surface_offset, normal_vs) / length(surface_offset)) {
        			rejection_bias *= sample_i == 0 ? 1 : 0;
        		}
            #endif

            // TODO: non-surface area metric

    #if !USE_APPROX_BRDF
                /*const float3 sample_origin_ws = sample_ray_ctx.ray_hit_ws();
                const float3 sample_hit_ws = sample_origin_ws + packed1.xyz;
                wi = mul(normalize(sample_hit_ws - view_ray_context.ray_hit_ws()), tangent_to_world);*/

                BrdfValue spec = specular_brdf.evaluate(wo, wi);

                // Note: looks closer to reference when comparing against metalness=1 albedo=1 PT reference.
                // As soon as the regular image is compared though. This term just happens to look
                // similar to the lack of Fresnel in the metalness=1 albedo=1 case.
                // float spec_weight = spec.value.x * max(0.0, wi.z);

                float spec_weight = calculate_luma(spec.value) * step(0.0, wi.z);
    #else
                float spec_weight;
                {
                    const float3 m = normalize(wo + wi);
                    const float cos_theta = m.z;
                    const float pdf_h_over_cos_theta = SpecularBrdf::ggx_ndf(a2, cos_theta);
                    //const float f = lerp(f0_grey, 1.0, approx_fresnel(wo, wi));
                    const float f = eval_fresnel_schlick(f0_grey, 1.0, dot(m, wi)).x;

                    spec_weight = f * pdf_h_over_cos_theta * step(0.0, wi.z) / max(1e-5, wo.z + wi.z);
                }
    #endif

                // Transform to projected surface area for the weight calculations below,
                // but don't change the encoded metrics of the input/output buffers.
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
                #endif

                float neighbor_sampling_pdf_psa = neighbor_sampling_pdf;
                neighbor_sampling_pdf /= to_psa_metric;

                if (spec_weight > 0 && rejection_bias > 0) {
                    //const float contrib_wt = geom_term * spec_weight * rejection_bias / neighbor_sampling_pdf;
                    const float contrib_wt = rejection_bias * step(0.0, wi.z) * spec_weight / neighbor_sampling_pdf;

                    float luma = calculate_luma(packed0.rgb);
                    //float reservoir_sel_rate = contrib_wt * sqrt(luma);
                    //float reservoir_sel_rate = rejection_bias * spec_weight * sqrt(luma);
                    //float reservoir_sel_rate = rejection_bias * sqrt(luma);
                    //float reservoir_sel_rate = exp2(-0.01 * error_adjusted_filter_size * sqrt(dot(sample_offset, sample_offset))) * contrib_wt * sqrt(luma);

                    float reservoir_sel_rate = contrib_wt * (0.01 + luma); // Note: 0 luma causes zero-probability samples which cause heavy bias in the ratio estimator
                    //float reservoir_sel_rate = contrib_wt * sqrt(0.01 + luma); // Note: 0 luma causes zero-probability samples which cause heavy bias in the ratio estimator
                    //float reservoir_sel_rate = luma;
                    //float reservoir_sel_rate = contrib_wt;
                    //float reservoir_sel_rate = 1;
                    //float reservoir_sel_rate = sample_i == 0 ? 1 : 0;

                    //float reservoir_sel_rate = sqrt(luma);
                    //reservoir_sel_rate *= packed1.a;
                    
                    const float pdf_contrib_wt = contrib_wt;

                    float reservoir_sel_prob = reservoir_sel_rate / (reservoir_rate_sum + reservoir_sel_rate);
                    float reservoir_sel_dart = uint_to_u01_float(hash1_mut(rng));
                    reservoir_rate_sum += reservoir_sel_rate;
                    reservoir_rate_weighted_sum += reservoir_sel_rate * pdf_contrib_wt;

                    reservoir_pdf_sum += neighbor_sampling_pdf_psa * pdf_contrib_wt;
                    reservoir_pdf_wt_sum += pdf_contrib_wt;
                    brdf_val_sum += luma;
                    valid_sample_count += 1;

                    if (reservoir_sel_prob > reservoir_sel_dart) {
                        reservoir_value0 = packed0;
                        //reservoir_value1 = float4(center_to_hit_vs.xyz, neighbor_sampling_pdf);
                        reservoir_value1 = float4(packed1.xyz, neighbor_sampling_pdf_psa);
                        reservoir_value2 = packed2;
                        reservoir_rate = reservoir_sel_rate;
                        reservoir_pdf = neighbor_sampling_pdf_psa;
                        picked_brdf_val = luma;
                    }
                }
            }
        }
    }

#if 0
    float combined_reservoir_rate =
        reservoir_value0.a *
        reservoir_rate / reservoir_rate_sum;
#else
    float combined_reservoir_rate = reservoir_rate_sum;
#endif

    float avg_brdf_val = brdf_val_sum / valid_sample_count;

    float avg_reservoir_rate = reservoir_rate_sum / valid_sample_count;

    float avg_reservoir_pdf = reservoir_pdf_sum / reservoir_pdf_wt_sum;
    float reservoir_rate_weighted_avg = reservoir_rate_weighted_sum / reservoir_pdf_wt_sum;

    //out_color.rgb = reservoir_value / max(1e-5, reservoir_rate * sample_count);
    //hit0_output_tex[px] = float4(reservoir_value0.rgb, reservoir_rate / reservoir_rate_weighted_avg);
//            hit0_output_tex[px] = float4(reservoir_value0.rgb, reservoir_rate / avg_reservoir_rate);
    //hit0_output_tex[px] = float4(reservoir_value0.rgb / (reservoir_rate / avg_reservoir_rate), reservoir_rate > 0 ? 1 : 0);
    //hit0_output_tex[px] = float4(reservoir_value0.rgb, reservoir_rate > 0 ? (picked_brdf_val / avg_brdf_val) : 0);
    hit0_output_tex[px] = float4(reservoir_value0.rgb, reservoir_pdf);

    //hit1_output_tex[px] = float4(reservoir_value1.rgb, reservoir_pdf);
    hit1_output_tex[px] = float4(reservoir_value1.rgb, reservoir_pdf * (reservoir_rate / reservoir_rate_sum));
    hit2_output_tex[px] = reservoir_value2;
    //hit1_output_tex[px] = float4(reservoir_value1.rgb, avg_reservoir_pdf);
}
