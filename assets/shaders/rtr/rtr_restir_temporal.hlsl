#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../inc/reservoir.hlsl"
#include "../ircache/bindings.hlsl"
#include "rtr_settings.hlsl"
#include "rtr_restir_pack_unpack.inc.hlsl"

#define RESTIR_RESERVOIR_W_CLAMP 1e20
#define RTR_RESTIR_BRDF_SAMPLING 1
#define USE_SPATIAL_TAPS_AT_LOW_M true
#define USE_RESAMPLING true

// Reject where the ray origin moves a lot
#define USE_TRANSLATIONAL_CLAMP true

// Fixes up some ellipses near contacts
#define USE_JACOBIAN_BASED_REJECTION true

// Causes some energy loss near contacts, but prevents
// ReSTIR from over-obsessing over them, and rendering
// tiny circles close to surfaces.
//
// TODO: This problem seems somewhat similar to what MIS fixes
// for light sampling; ReSTIR here is similar in behavior to a light sampling technique,
// and it similarly becomes bad close to the source, where BRDF sampling
// works perfectly fine. Maybe we can tackle it in a similar way.
#define USE_DISTANCE_BASED_M_CLAMP !true

#define USE_REPROJECTION_SEARCH true


[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float3> half_view_normal_tex;
[[vk::binding(2)]] Texture2D<float> depth_tex;
[[vk::binding(3)]] Texture2D<float4> candidate0_tex;
[[vk::binding(4)]] Texture2D<float4> candidate1_tex;
[[vk::binding(5)]] Texture2D<float4> candidate2_tex;
[[vk::binding(6)]] Texture2D<float4> irradiance_history_tex;
[[vk::binding(7)]] Texture2D<float4> ray_orig_history_tex;
[[vk::binding(8)]] Texture2D<float4> ray_history_tex;
[[vk::binding(9)]] Texture2D<uint> rng_history_tex;
[[vk::binding(10)]] Texture2D<uint2> reservoir_history_tex;
[[vk::binding(11)]] Texture2D<float4> reprojection_tex;
[[vk::binding(12)]] Texture2D<float4> hit_normal_history_tex;
[[vk::binding(13)]] RWTexture2D<float4> irradiance_out_tex;
[[vk::binding(14)]] RWTexture2D<float4> ray_orig_output_tex;
[[vk::binding(15)]] RWTexture2D<float4> ray_output_tex;
[[vk::binding(16)]] RWTexture2D<uint> rng_output_tex;
[[vk::binding(17)]] RWTexture2D<float4> hit_normal_output_tex;
[[vk::binding(18)]] RWTexture2D<uint2> reservoir_out_tex;
[[vk::binding(19)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

static const float SKY_DIST = 1e4;

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

struct TraceResult {
    float3 out_value;
    float3 hit_normal_ws;
    float3 hit_vs;
    float hit_t;
    float pdf;
    float cos_theta;
};

TraceResult do_the_thing(uint2 px, float3 primary_hit_normal) {
    const float4 hit0 = candidate0_tex[px];
    const float4 hit1 = candidate1_tex[px];
    const float4 hit2 = candidate2_tex[px];

    TraceResult result;
    result.out_value = hit0.rgb;
    result.pdf = min(hit1.a, RTR_RESTIR_MAX_PDF_CLAMP);
    result.cos_theta = rtr_decode_cos_theta_from_fp16(hit0.a);
    result.hit_vs = hit1.xyz;
    result.hit_t = length(hit1.xyz);
    result.hit_normal_ws = direction_view_to_world(hit2.xyz);
    return result;
}

float4 decode_hit_normal_and_dot(float4 val) {
    return float4(val.xyz * 2 - 1, val.w);
}

float4 encode_hit_normal_and_dot(float4 val) {
    return float4(val.xyz * 0.5 + 0.5, val.w);
}

// Sometimes the best reprojection of the point-sampled reservoir data is not exactly at the pixel we're looking at.
// We need to inspect a tiny neighborhood around the point. This helps with shimmering edges, but also upon
// movement, since we can't linearly sample the reservoir data.
void find_best_reprojection_in_neighborhood(float2 base_px, inout int2 best_px, float3 refl_ray_origin_ws, bool wide) {
    float best_dist = 1e10;

    const float2 clip_scale = frame_constants.view_constants.clip_to_view._m00_m11;
    const float2 offset_scale = float2(1, -1) * -2 * clip_scale * gbuffer_tex_size.zw;

    const float3 look_direction = direction_view_to_world(float3(0, 0, -1));

    {
        const float z_offset = dot(look_direction, refl_ray_origin_ws - get_eye_position());

        // Subtract the subsample XY offset from the comparison position.
        // This will prevent the search from constantly re-shuffling pixels due to the sub-sample jitters.
        refl_ray_origin_ws
            += direction_view_to_world(float3(float2(HALFRES_SUBSAMPLE_OFFSET) * offset_scale * z_offset, 0));
    }

    const int start_coord = select(wide, -1, 0);
    for (int y = start_coord; y <= 1; ++y) {
        for (int x = start_coord; x <= 1; ++x) {
            int2 spx = floor(base_px + float2(x, y));

            RtrRestirRayOrigin ray_orig = RtrRestirRayOrigin::from_raw(ray_orig_history_tex[spx]);
            float3 orig = ray_orig.ray_origin_eye_offset_ws + get_prev_eye_position();
            uint2 orig_jitter = hi_px_subpixels[ray_orig.frame_index_mod4];

            {
                const float z_offset = dot(look_direction, orig);

                // Similarly subtract the subsample XY offset that the ray was traced with.
                orig += direction_view_to_world(float3(float2(orig_jitter) * offset_scale * z_offset, 0));
            }

            float d = length(orig - refl_ray_origin_ws);

            if (d < best_dist) {
                best_dist = d;
                best_px = spx;
            }
        }
    }
}

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID) {
    const int2 hi_px_offset = HALFRES_SUBSAMPLE_OFFSET;
    const uint2 hi_px = px * 2 + hi_px_offset;
    
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        irradiance_out_tex[px] = float4(0.0.xxx, -SKY_DIST);
        hit_normal_output_tex[px] = 0.0.xxxx;
        reservoir_out_tex[px] = 0;
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const float3 normal_vs = half_view_normal_tex[px];
    const float3 normal_ws = direction_view_to_world(normal_vs);

    float local_normal_flatness = 1; {
        const int k = 1;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float3 sn_vs = half_view_normal_tex[px + int2(x, y)];
                local_normal_flatness *= saturate(dot(normal_vs, sn_vs));
            }
        }
    }

    float reprojection_neighborhood_stability = 1;
    {
        for (int y = 0; y <= 1; ++y) {
            for (int x = 0; x <= 1; ++x) {
                float r = reprojection_tex[px * 2 + int2(x, y)].z;
                reprojection_neighborhood_stability *= r;
            }
        }
    }

#if RTR_USE_TIGHTER_RAY_BIAS
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_biased_depth(uv, depth);
    const float3 refl_ray_origin_ws = view_ray_context.biased_secondary_ray_origin_ws_with_normal(normal_ws);
#else
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);
    const float3 refl_ray_origin_ws = view_ray_context.biased_secondary_ray_origin_ws();
#endif

    const float3 refl_ray_origin_vs = position_world_to_view(refl_ray_origin_ws);

    const float3x3 tangent_to_world = build_orthonormal_basis(normal_ws);
    float3 outgoing_dir = float3(0, 0, 1);

    uint rng = hash3(uint3(px, frame_constants.frame_index));
    float3 wo = mul(-normalize(view_ray_context.ray_dir_ws()), tangent_to_world);

    // Hack for shading normals facing away from the outgoing ray's direction:
    // We flip the outgoing ray along the shading normal, so that the reflection's curvature
    // continues, albeit at a lower rate.
    if (wo.z < 0.0) {
        wo.z *= -0.25;
        wo = normalize(wo);
    }

    const float4 gbuffer_packed = gbuffer_tex[hi_px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    SpecularBrdf specular_brdf;
    {
        LayeredBrdf layered_brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);
        specular_brdf = layered_brdf.specular_brdf;
    }
    const float a2 = max(RTR_ROUGHNESS_CLAMP, gbuffer.roughness) * max(RTR_ROUGHNESS_CLAMP, gbuffer.roughness);

    // TODO: use
    float3 light_radiance = 0.0.xxx;

    float p_q_sel = 0;
    float pdf_sel = 0;
    float cos_theta = 0;
    float3 irradiance_sel = 0;
    float4 ray_orig_sel = 0;
    float3 ray_hit_sel_ws = 1;
    float3 hit_normal_sel = 1;
    uint rng_sel = rng_output_tex[px];

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState::create();
    Reservoir1spp reservoir = Reservoir1spp::create();
    const uint reservoir_payload = px.x | (px.y << 16);

    reservoir.payload = reservoir_payload;

    {
        TraceResult result = do_the_thing(px, normal_ws);

        if (result.pdf > 0) {
            outgoing_dir = normalize(result.hit_vs);

            float3 wi = normalize(mul(outgoing_dir, tangent_to_world));

            const float p_q = p_q_sel = 1
                * max(1e-3, sRGB_to_luminance(result.out_value))
                #if !RTR_RESTIR_BRDF_SAMPLING
                    * max(0, dot(outgoing_dir, normal_ws))
                #endif
                //* sRGB_to_luminance(specular_brdf.evaluate(wo, wi).value)
                * result.pdf
                ;

            const float inv_pdf_q = 1.0 / result.pdf;

            pdf_sel = result.pdf;
            cos_theta = result.cos_theta;
            
            irradiance_sel = result.out_value;

            RtrRestirRayOrigin ray_orig;
            // Note: needs patching up by the eye pos later.
            ray_orig.ray_origin_eye_offset_ws = refl_ray_origin_ws;
            ray_orig.roughness = gbuffer.roughness;
            ray_orig.frame_index_mod4 = frame_constants.frame_index & 3;

            ray_orig_sel = ray_orig.to_raw();

            ray_hit_sel_ws = result.hit_vs + refl_ray_origin_ws;

            hit_normal_sel = result.hit_normal_ws;

            if (p_q * inv_pdf_q > 0) {
                reservoir.init_with_stream(p_q, inv_pdf_q, stream_state, reservoir_payload);
            }
        }
    }

    //const bool use_resampling = false;
    const bool use_resampling = USE_RESAMPLING;
    const float4 center_reproj = reprojection_tex[hi_px];

    if (use_resampling) {
        const float ang_offset = ((frame_constants.frame_index + 7) * 11) % 32 * M_TAU;

        for (uint sample_i = 0; sample_i < select((USE_SPATIAL_TAPS_AT_LOW_M && center_reproj.z < 1.0), 5, 1) && stream_state.M_sum < RTR_RESTIR_TEMPORAL_M_CLAMP; ++sample_i) {
        //for (uint sample_i = 0; sample_i < 1; ++sample_i) {
            const float ang = (sample_i + ang_offset) * GOLDEN_ANGLE;
            const float rpx_offset_radius = sqrt(
                float(((sample_i - 1) + frame_constants.frame_index) & 3) + 1
            ) * clamp(8 - stream_state.M_sum, 1, 7); // TODO: keep high in noisy situations
            //) * 7;
            const float2 reservoir_px_offset_base = float2(
                cos(ang), sin(ang)
            ) * rpx_offset_radius;

            const int2 rpx_offset =
                select(sample_i == 0
                , int2(0, 0)
                , int2(reservoir_px_offset_base))
                ;

            float4 reproj = reprojection_tex[hi_px + rpx_offset * 2];

            // Can't use linear interpolation.

            const float2 reproj_px_flt = px + gbuffer_tex_size.xy * reproj.xy / 2;

            int2 reproj_px;
            

            {
                const float2 base_px = px + gbuffer_tex_size.xy * reproj.xy / 2;
                int2 best_px = floor(base_px + 0.5);

                if (USE_REPROJECTION_SEARCH) {
                    #if USE_HALFRES_SUBSAMPLE_JITTERING
                        if (reprojection_neighborhood_stability >= 1) {
                            // If the neighborhood is stable, we can do a tiny search to find a reprojection
                            // that has the best chance of keeping reservoirs alive.
                            // Only bother if there's any motion. If we do this when there's no motion,
                            // we may end up creating too much correlation between pixels by shuffling them around.

                            if (any(abs(gbuffer_tex_size.xy * reproj.xy) > 0.1)) {
                                find_best_reprojection_in_neighborhood(base_px, best_px, refl_ray_origin_ws.xyz, false);
                            }
                        } else {
                            // The neighborhood is not stable. Shimmering or moving edges.
                            // Do a more aggressive search.

                            find_best_reprojection_in_neighborhood(base_px, best_px, refl_ray_origin_ws.xyz, true);
                        }
                    #else
                        // If subsample jittering is disabled, we only ever need the tiny search

                        if (any(abs(gbuffer_tex_size.xy * reproj.xy) > 0.1)) {
                            find_best_reprojection_in_neighborhood(base_px, best_px, refl_ray_origin_ws.xyz, false);
                        }
                    #endif
                }

                reproj_px = best_px;
            }

            const int2 rpx = reproj_px + rpx_offset;
            const uint2 rpx_hi = rpx * 2 + hi_px_offset;

            const float3 sample_normal_vs = half_view_normal_tex[rpx];
            // Note: also doing this for sample 0, as under extreme aliasing,
            // we can easily get bad samples in.
            if (dot(sample_normal_vs, normal_vs) < 0.7) {
                //continue;
            }

            Reservoir1spp r = Reservoir1spp::from_raw(reservoir_history_tex[rpx]);
            const uint2 spx = reservoir_payload_to_px(r.payload);

            const float2 sample_uv = get_uv(rpx_hi, gbuffer_tex_size);
            const float4 prev_ray_orig_and_roughness = ray_orig_history_tex[spx] + float4(get_prev_eye_position(), 0);

            // Reject disocclusions
            if (length_squared(refl_ray_origin_ws - prev_ray_orig_and_roughness.xyz) > 0.05 * refl_ray_origin_vs.z * refl_ray_origin_vs.z) {
                continue;
            }

            const float4 prev_irrad_and_cos_theta =
                irradiance_history_tex[spx]
                * float4((frame_constants.pre_exposure_delta).xxx, 1);

            const float3 prev_irrad = prev_irrad_and_cos_theta.rgb;
            const float prev_cos_theta = rtr_decode_cos_theta_from_fp16(prev_irrad_and_cos_theta.a);

            const float4 sample_hit_ws_and_pdf_packed = ray_history_tex[spx];
            const float prev_pdf = sample_hit_ws_and_pdf_packed.a;

            const float3 sample_hit_ws = sample_hit_ws_and_pdf_packed.xyz + prev_ray_orig_and_roughness.xyz;
            //const float3 prev_dir_to_sample_hit_unnorm_ws = sample_hit_ws - sample_ray_ctx.ray_hit_ws();
            //const float3 prev_dir_to_sample_hit_ws = normalize(prev_dir_to_sample_hit_unnorm_ws);
            const float prev_dist = length(sample_hit_ws_and_pdf_packed.xyz);
            //const float prev_dist = length(prev_dir_to_sample_hit_unnorm_ws);

            // Note: needs `spx` since `hit_normal_history_tex` is not reprojected.
            const float4 sample_hit_normal_ws_dot = decode_hit_normal_and_dot(hit_normal_history_tex[spx]);

            const float3 dir_to_sample_hit_unnorm = sample_hit_ws - refl_ray_origin_ws;
            const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
            const float3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);
            
            // From the ReSTIR paper:
            // With temporal reuse, the number of candidates M contributing to the
            // pixel can in theory grow unbounded, as each frame always combines
            // its reservoir with the previous frame’s. This causes (potentially stale)
            // temporal samples to be weighted disproportionately high during
            // resampling. To fix this, we simply clamp the previous frame’s M
            // to at most 20× of the current frame’s reservoir’s M

            r.M = min(r.M, RTR_RESTIR_TEMPORAL_M_CLAMP);

            const float3 wi = normalize(mul(dir_to_sample_hit, tangent_to_world));

            if (USE_TRANSLATIONAL_CLAMP) {
                //const float3 current_wo = normalize(ViewRayContext::from_uv(uv).ray_dir_vs());
                //const float3 prev_wo = normalize(ViewRayContext::from_uv(uv + center_reproj.xy).ray_dir_vs());

                // TODO: take object motion into account too
                const float3 current_wo = normalize(view_ray_context.ray_hit_ws() - get_eye_position());
                const float3 prev_wo = normalize(view_ray_context.ray_hit_ws() - get_prev_eye_position());

                const float wo_dot = saturate(dot(current_wo, prev_wo));

                const float wo_similarity =
                    pow(saturate(SpecularBrdf::ggx_ndf_0_1(max(3e-5, a2), wo_dot)), 64);

                float mult = lerp(wo_similarity, 1, smoothstep(0.05, 0.5, sqrt(gbuffer.roughness)));
                
                // Don't bother if the surface is bumpy. The lag is hard to see then,
                // and we'd just end up introducing aliasing on small features.
                mult = lerp(1.0, mult, local_normal_flatness);

                r.M *= mult;
            }

            float p_q = 1;
            p_q *= max(1e-3, sRGB_to_luminance(prev_irrad.rgb));
            #if !RTR_RESTIR_BRDF_SAMPLING
                p_q *= max(0, dot(dir_to_sample_hit, normal_ws));
            #else
                p_q *= step(0, dot(dir_to_sample_hit, normal_ws));
            #endif
            p_q *= prev_pdf;
            //p_q *= sRGB_to_luminance(specular_brdf.evaluate(wo, wi).value);

            float visibility = 1;
            float jacobian = 1;

            // Note: needed for sample 0 due to temporal jitter.
            if (true) {
                // Distance falloff. Needed to avoid leaks.
                jacobian *= clamp(prev_dist / dist_to_sample_hit, 1e-4, 1e4);
                jacobian *= jacobian;

                // N of hit dot -L. Needed to avoid leaks.
                jacobian *=
                    max(0.0, -dot(sample_hit_normal_ws_dot.xyz, dir_to_sample_hit))
                    / max(1e-5, sample_hit_normal_ws_dot.w);
                    /// max(1e-5, -dot(sample_hit_normal_ws_dot.xyz, prev_dir_to_sample_hit_ws));

                #if RTR_RESTIR_BRDF_SAMPLING
                    // N dot L. Useful for normal maps, micro detail.
                    // The min(const, _) should not be here, but it prevents fireflies and brightening of edges
                    // when we don't use a harsh normal cutoff to exchange reservoirs with.
                    //jacobian *= min(1.2, max(0.0, prev_irrad.a) / dot(dir_to_sample_hit, center_normal_ws));
                    //jacobian *= max(0.0, prev_irrad.a) / dot(dir_to_sample_hit, center_normal_ws);
                #endif
            }

            // Fixes boiling artifacts near edges. Unstable jacobians,
            // but also effectively reduces reliance on reservoir exchange
            // in tight corners, which is desirable since the well-distributed
            // raw samples thrown at temporal filters will do better.
            if (USE_JACOBIAN_BASED_REJECTION) {
                const float JACOBIAN_REJECT_THRESHOLD = lerp(1.1, 4.0, gbuffer.roughness * gbuffer.roughness);
                if (!(jacobian < JACOBIAN_REJECT_THRESHOLD && jacobian > 1.0 / JACOBIAN_REJECT_THRESHOLD)) {
                    continue;
                    //r.M *= pow(saturate(1 - max(jacobian, 1.0 / jacobian) / JACOBIAN_REJECT_THRESHOLD), 4.0);
                }
            }

            if (USE_DISTANCE_BASED_M_CLAMP) {
                // ReSTIR tends to produce firflies near contacts.
                // This is a hack to reduce the effect while I figure out a better solution.
                // HACK: reduce M close to surfaces.
                //
                // Note: This causes ReSTIR to be less effective, and can manifest
                // as darkening in corners. Since it's mostly useful for smoother surfaces,
                // fade it out when they're rough.
                const float dist_to_hit_vs_scaled =
                    dist_to_sample_hit
                    / -refl_ray_origin_vs.z
                    * frame_constants.view_constants.view_to_clip[1][1];
                {
                    float dist2 = dot(ray_hit_sel_ws - refl_ray_origin_ws, ray_hit_sel_ws - refl_ray_origin_ws);
                    dist2 = min(dist2, 2 * dist_to_hit_vs_scaled * dist_to_hit_vs_scaled);
                    r.M = min(r.M, RTR_RESTIR_TEMPORAL_M_CLAMP * lerp(saturate(50.0 * dist2), 1.0, gbuffer.roughness * gbuffer.roughness));
                }
            }

            // We're not recalculating the PDF-based factor of p_q,
            // so it needs measure adjustment.
            p_q *= jacobian;

            // TODO: consider ray-marching for occlusion

            if (reservoir.update_with_stream(
                r, p_q, visibility,
                stream_state, reservoir_payload, rng
            )) {
                outgoing_dir = dir_to_sample_hit;
                p_q_sel = p_q;
                pdf_sel = prev_pdf;
                cos_theta = prev_cos_theta;
                irradiance_sel = prev_irrad.rgb;

// TODO: was `refl_ray_origin_ws`; what should it be?
                //ray_orig_sel = refl_ray_origin_ws;
                ray_orig_sel = prev_ray_orig_and_roughness;

                ray_hit_sel_ws = sample_hit_ws;
                hit_normal_sel = sample_hit_normal_ws_dot.xyz;

                rng_sel = rng_history_tex[spx];
            }
        }

        reservoir.finish_stream(stream_state);
        reservoir.W = min(reservoir.W, RESTIR_RESERVOIR_W_CLAMP);
    }

    RayDesc outgoing_ray;
    outgoing_ray.Direction = outgoing_dir;
    outgoing_ray.Origin = refl_ray_origin_ws;
    outgoing_ray.TMin = 0;

    const float4 hit_normal_ws_dot = float4(hit_normal_sel, -dot(hit_normal_sel, outgoing_ray.Direction));

    irradiance_out_tex[px] = float4(irradiance_sel, rtr_encode_cos_theta_for_fp16(cos_theta));
    // Note: relies on the `xyz` being directly encoded by `RtrRestirRayOrigin`
    ray_orig_output_tex[px] = float4(ray_orig_sel.xyz - get_eye_position(), ray_orig_sel.w);
    hit_normal_output_tex[px] = encode_hit_normal_and_dot(hit_normal_ws_dot);
    ray_output_tex[px] = float4(ray_hit_sel_ws - ray_orig_sel.xyz, pdf_sel);
    rng_output_tex[px] = rng_sel;
    reservoir_out_tex[px] = reservoir.as_raw();
}
