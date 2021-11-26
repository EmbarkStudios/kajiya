#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../inc/reservoir.hlsl"
#include "../surfel_gi/bindings.hlsl"
#include "rtr_settings.hlsl"

#define RESTIR_TEMPORAL_M_CLAMP 8.0
#define RESTIR_RESERVOIR_W_CLAMP 1e20
#define RESTIR_USE_PATH_VALIDATION !true
#define RTR_RESTIR_BRDF_SAMPLING 1

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float3> half_view_normal_tex;
[[vk::binding(2)]] Texture2D<float> depth_tex;
[[vk::binding(3)]] Texture2D<float4> candidate0_tex;
[[vk::binding(4)]] Texture2D<float4> candidate1_tex;
[[vk::binding(5)]] Texture2D<float4> candidate2_tex;
[[vk::binding(6)]] Texture2D<float4> irradiance_history_tex;
[[vk::binding(7)]] Texture2D<float4> ray_orig_history_tex;
[[vk::binding(8)]] Texture2D<float4> ray_history_tex;
[[vk::binding(9)]] Texture2D<float4> reservoir_history_tex;
[[vk::binding(10)]] Texture2D<float4> reprojection_tex;
[[vk::binding(11)]] Texture2D<float4> hit_normal_history_tex;
//[[vk::binding(11)]] Texture2D<float4> candidate_history_tex;
[[vk::binding(12)]] RWTexture2D<float4> irradiance_out_tex;
[[vk::binding(13)]] RWTexture2D<float4> ray_orig_output_tex;
[[vk::binding(14)]] RWTexture2D<float4> ray_output_tex;
[[vk::binding(15)]] RWTexture2D<float4> hit_normal_output_tex;
[[vk::binding(16)]] RWTexture2D<float4> reservoir_out_tex;
//[[vk::binding(17)]] RWTexture2D<float4> candidate_out_tex;
[[vk::binding(17)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

static const float SKY_DIST = 1e4;

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

struct TraceResult {
    float3 out_value;
    float3 hit_normal_ws;
    float3 hit_pos_vs;
    float hit_t;
    float pdf;
    float ratio_estimator_factor;
    bool prev_sample_valid;
};

TraceResult do_the_thing(uint2 px, float3 primary_hit_normal) {
    const float4 hit0 = candidate0_tex[px];
    const float4 hit1 = candidate1_tex[px];
    const float4 hit2 = candidate2_tex[px];

    TraceResult result;
    result.out_value = hit0.rgb;
    result.pdf = min(hit1.a, RTR_RESTIR_MAX_PDF_CLAMP);
    result.ratio_estimator_factor = hit0.a;
    #if !RTR_RAY_HIT_STORED_AS_POSITION
        todo
    #endif
    result.hit_pos_vs = hit1.xyz;
    result.hit_t = length(hit1.xyz);
    result.hit_normal_ws = direction_view_to_world(hit2.xyz);
    result.prev_sample_valid = true;
    return result;
}

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID) {
    const uint2 hi_px_subpixels[4] = {
        uint2(0, 0),
        uint2(1, 1),
        uint2(1, 0),
        uint2(0, 1),
    };

    const int2 hi_px_offset = hi_px_subpixels[frame_constants.frame_index & 3];
    const uint2 hi_px = px * 2 + hi_px_offset;
    
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        irradiance_out_tex[px] = float4(0.0.xxx, -SKY_DIST);
        hit_normal_output_tex[px] = 0.0.xxxx;
        reservoir_out_tex[px] = 0.0.xxxx;
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);
    const float3 normal_vs = half_view_normal_tex[px];
    const float3 normal_ws = direction_view_to_world(normal_vs);
    const float3x3 tangent_to_world = build_orthonormal_basis(normal_ws);
    const float3 refl_ray_origin = view_ray_context.biased_secondary_ray_origin_ws();
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

    // TODO: use
    float3 light_radiance = 0.0.xxx;

    float p_q_sel = 0;
    float pdf_sel = 0;
    float ratio_estimator_factor = 0;
    uint2 src_px_sel = px;
    float3 irradiance_sel = 0;
    float3 ray_orig_sel = 0;
    float3 ray_hit_sel = 1;
    float3 hit_normal_sel = 1;
    uint sel_valid_sample_idx = 0;
    bool prev_sample_valid = false;

    Reservoir1spp reservoir = Reservoir1spp::create();
    const uint reservoir_payload = px.x | (px.y << 16);

    reservoir.payload = reservoir_payload;

    {
        TraceResult result = do_the_thing(px, normal_ws);
        outgoing_dir = normalize(position_view_to_world(result.hit_pos_vs) - refl_ray_origin);
        float3 wi = normalize(mul(outgoing_dir, tangent_to_world));

        const float p_q = p_q_sel = 1
            * max(1e-3, calculate_luma(result.out_value))
            #if !RTR_RESTIR_BRDF_SAMPLING
                * max(0, dot(outgoing_dir, normal_ws))
            #endif
            //* calculate_luma(specular_brdf.evaluate(wo, wi).value)
            * result.pdf
            ;

        const float inv_pdf_q = 1.0 / result.pdf;

        pdf_sel = result.pdf;
        ratio_estimator_factor = result.ratio_estimator_factor;//clamp(result.inv_pdf, 1e-5, 1e5);
        irradiance_sel = result.out_value;
        ray_orig_sel = refl_ray_origin;
        ray_hit_sel = position_view_to_world(result.hit_pos_vs);
        hit_normal_sel = result.hit_normal_ws;
        prev_sample_valid = result.prev_sample_valid;

        if (p_q * inv_pdf_q > 0) {
            reservoir.payload = reservoir_payload;
            reservoir.w_sum = p_q * inv_pdf_q;
            reservoir.M = 1;
            reservoir.W = inv_pdf_q;
        }

        //float rl = lerp(candidate_history_tex[px].y, sqrt(result.hit_t), 0.05);
        //candidate_out_tex[px] = float4(sqrt(result.hit_t), rl, 0, 0);
    }

    //const bool use_resampling = false;
    const bool use_resampling = prev_sample_valid && true;

    // 1 (center) plus offset samples
    const uint MAX_RESOLVE_SAMPLE_COUNT = 5;

    int2 sample_offsets[4] = {
        int2(1, 0),
        int2(0, 1),
        int2(-1, 0),
        int2(0, -1),
    };

    const float rt_invalidity = 0;//sqrt(rt_invalidity_tex[px]);

    if (use_resampling) {
        float M_sum = reservoir.M;

        uint valid_sample_count = 0;
        const float ang_offset = ((frame_constants.frame_index + 7) * 11) % 32 * M_TAU;

        //for (uint sample_i = 0; sample_i < 1 && M_sum < RESTIR_TEMPORAL_M_CLAMP * 1.25; ++sample_i) {
        for (uint sample_i = 0; sample_i < 1; ++sample_i) {
            const float ang = (sample_i + ang_offset) * GOLDEN_ANGLE;
            const float rpx_offset_radius = sqrt(
                float(((sample_i - 1) + frame_constants.frame_index) & 3) + 1
            ) * clamp(8 - M_sum, 1, 7); // TODO: keep high in noisy situations
            //) * 7;
            const float2 reservoir_px_offset_base = float2(
                cos(ang), sin(ang)
            ) * rpx_offset_radius;

            const int2 rpx_offset =
                sample_i == 0
                ? 0
                //: sample_offsets[((sample_i - 1) + frame_constants.frame_index) & 3];
                : int2(reservoir_px_offset_base)
                ;

            const float4 reproj = reprojection_tex[hi_px + rpx_offset * 2];

            // Can't use linear interpolation, but we can interpolate stochastically instead
            //const float2 reproj_rand_offset = float2(uint_to_u01_float(hash1_mut(rng)), uint_to_u01_float(hash1_mut(rng))) - 0.5;
            // Or not at all.
            const float2 reproj_rand_offset = 0.0;

            int2 reproj_px = floor((
                sample_i == 0
                ? px
                // My poor approximation of permutation sampling.
                // https://twitter.com/more_fps/status/1457749362025459715
                //
                // When applied everywhere, it does nicely reduce noise, but also makes the GI less reactive
                // since we're effectively increasing the lifetime of the most attractive samples.
                // Where it does come in handy though is for boosting convergence rate for newly revealed
                // locations.
                : ((px + /*prev_*/rpx_offset) ^ 3)) + gbuffer_tex_size.xy * reproj.xy / 2 + reproj_rand_offset + 0.5);
            //int2 reproj_px = floor(px + gbuffer_tex_size.xy * reproj.xy / 2 + reproj_rand_offset + 0.5);

            const int2 rpx = reproj_px + rpx_offset;
            const uint2 rpx_hi = rpx * 2 + hi_px_offset;

            const float3 sample_normal_vs = half_view_normal_tex[rpx];
            // Note: also doing this for sample 0, as under extreme aliasing,
            // we can easily get bad samples in.
            if (dot(sample_normal_vs, normal_vs) < 0.7) {
                continue;
            }

            Reservoir1spp r = Reservoir1spp::from_raw(reservoir_history_tex[rpx]);
            const uint2 spx = reservoir_payload_to_px(r.payload);

            const float2 sample_uv = get_uv(rpx_hi, gbuffer_tex_size);
            const float sample_depth = depth_tex[rpx_hi];
            
            // Note: also doing this for sample 0, as under extreme aliasing,
            // we can easily get bad samples in.
            if (0 == sample_depth) {
                continue;
            }

            // TODO: some more rejection based on the reprojection map.
            // This one is not enough ("battle", buttom of tower).
            if (inverse_depth_relative_diff(depth, sample_depth) > 0.2 || reproj.z == 0) {
                continue;
            }

            //const ViewRayContext sample_ray_ctx = ViewRayContext::from_uv_and_depth(sample_uv, sample_depth);
            const float4 sample_hit_ws_and_dist = ray_history_tex[spx]/* + float4(get_prev_eye_position(), 0.0)*/;
            const float3 sample_hit_ws = sample_hit_ws_and_dist.xyz;
            //const float3 prev_dir_to_sample_hit_unnorm_ws = sample_hit_ws - sample_ray_ctx.ray_hit_ws();
            //const float3 prev_dir_to_sample_hit_ws = normalize(prev_dir_to_sample_hit_unnorm_ws);
            const float prev_dist = ray_orig_history_tex[spx].w;
            //const float prev_dist = length(prev_dir_to_sample_hit_unnorm_ws);

            // Note: needs `spx` since `hit_normal_history_tex` is not reprojected.
            const float4 sample_hit_normal_ws_dot = hit_normal_history_tex[spx];

            /*if (sample_i > 0 && !(prev_dist > 1e-4)) {
                continue;
            }*/

            const float3 dir_to_sample_hit_unnorm = sample_hit_ws - refl_ray_origin;
            const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
            const float3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);

            // Note: also doing this for sample 0, as under extreme aliasing,
            // we can easily get bad samples in.
            if (dot(dir_to_sample_hit, normal_ws) < 1e-3) {
                continue;
            }
            
            const float4 prev_irrad_pdf = irradiance_history_tex[spx];
            const float3 prev_irrad = prev_irrad_pdf.rgb;

            // From the ReSTIR paper:
            // With temporal reuse, the number of candidates M contributing to the
            // pixel can in theory grow unbounded, as each frame always combines
            // its reservoir with the previous frame’s. This causes (potentially stale)
            // temporal samples to be weighted disproportionately high during
            // resampling. To fix this, we simply clamp the previous frame’s M
            // to at most 20× of the current frame’s reservoir’s M

            r.M = min(r.M, RESTIR_TEMPORAL_M_CLAMP * lerp(1.0, 0.25, rt_invalidity));
            //r.M = min(r.M, RESTIR_TEMPORAL_M_CLAMP);

            const float3 wi = normalize(mul(dir_to_sample_hit, tangent_to_world));

            float p_q = 1;
            p_q *= max(1e-3, calculate_luma(prev_irrad.rgb));
            #if !RTR_RESTIR_BRDF_SAMPLING
                p_q *= max(0, dot(dir_to_sample_hit, normal_ws));
            #else
                p_q *= step(0, dot(dir_to_sample_hit, normal_ws));
            #endif
            p_q *= prev_irrad_pdf.w;
            //p_q *= calculate_luma(specular_brdf.evaluate(wo, wi).value);

            float visibility = 1;
            float jacobian = 1;

            // Note: needed for sample 0 due to temporal jitter.
            //if (sample_i > 0)
            {
                // Distance falloff. Needed to avoid leaks.
                jacobian *= clamp(prev_dist, 1e-4, 1e4) / clamp(dist_to_sample_hit, 1e-4, 1e4);
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

                    //jacobian /= clamp(specular_brdf.evaluate(wo, wi).pdf, 1e-5, 1e5) * prev_irrad_pdf.w;
                #endif
            }

            // Raymarch to check occlusion
            if (sample_i > 0) {
                const ViewRayContext sample_ray_ctx = ViewRayContext::from_uv_and_depth(sample_uv, sample_depth);
                const float3 sample_origin_vs = sample_ray_ctx.ray_hit_vs();
        		const float3 surface_offset_vs = sample_origin_vs - view_ray_context.ray_hit_vs();

                // TODO: finish the derivations, don't perspective-project for every sample.

                const float3 raymarch_dir_unnorm_ws = sample_hit_ws - view_ray_context.ray_hit_ws();
                const float3 raymarch_end_ws =
                    view_ray_context.ray_hit_ws()
                    // TODO: what's a good max distance to raymarch? Probably need to project some stuff
                    + raymarch_dir_unnorm_ws * min(1.0, length(surface_offset_vs) / length(raymarch_dir_unnorm_ws));

                const float2 raymarch_end_uv = cs_to_uv(position_world_to_clip(raymarch_end_ws).xy);
                const float2 raymarch_len_px = (raymarch_end_uv - uv) * gbuffer_tex_size.xy;

                const uint MIN_PX_PER_STEP = 2;
                const uint MAX_TAPS = 2;

                const int k_count = min(MAX_TAPS, int(floor(length(raymarch_len_px) / MIN_PX_PER_STEP)));

                // Depth values only have the front; assume a certain thickness.
                const float Z_LAYER_THICKNESS = 0.05;

                for (int k = 0; k < k_count; ++k) {
                    const float t = (k + 0.5) / k_count;
                    const float3 interp_pos_ws = lerp(view_ray_context.ray_hit_ws(), raymarch_end_ws, t);
                    const float3 interp_pos_cs = position_world_to_clip(interp_pos_ws);
                    const float depth_at_interp = depth_tex.SampleLevel(sampler_nnc, cs_to_uv(interp_pos_cs.xy), 0);
                    if (depth_at_interp > interp_pos_cs.z) {
                        visibility *= smoothstep(0, Z_LAYER_THICKNESS, inverse_depth_relative_diff(interp_pos_cs.z, depth_at_interp));
                    }
                }
    		}

            M_sum += r.M;
            if (reservoir.update(p_q * r.W * r.M * jacobian * visibility, reservoir_payload, rng)) {
                outgoing_dir = dir_to_sample_hit;
                p_q_sel = p_q;
                pdf_sel = prev_irrad_pdf.w;
                ratio_estimator_factor = ray_history_tex[spx].w;//1.0 / specular_brdf.evaluate(wo, wi).value.x;
                src_px_sel = rpx;
                irradiance_sel = prev_irrad.rgb;
                ray_orig_sel = ray_orig_history_tex[spx].xyz;
                ray_hit_sel = sample_hit_ws;
                hit_normal_sel = sample_hit_normal_ws_dot.xyz;
                sel_valid_sample_idx = valid_sample_count;
            }
        }

        valid_sample_count = max(valid_sample_count, 1);

        reservoir.M = M_sum;
        reservoir.W = (1.0 / max(1e-5, p_q_sel)) * (reservoir.w_sum / max(1e-5, reservoir.M));

        // TODO: find out if we can get away with this:
        reservoir.W = min(reservoir.W, RESTIR_RESERVOIR_W_CLAMP);
    }

    RayDesc outgoing_ray;
    outgoing_ray.Direction = outgoing_dir;
    outgoing_ray.Origin = refl_ray_origin;
    outgoing_ray.TMin = 0;

    //TraceResult result = do_the_thing(px, rng, outgoing_ray, gbuffer);

    const float4 hit_normal_ws_dot = float4(hit_normal_sel, -dot(hit_normal_sel, outgoing_ray.Direction));

    /*if (any(src_px_sel != px)) {
        const uint2 spx = src_px_sel;
        const float4 hit_normal_ws_dot = hit_normal_history_tex[spx];
        jacobian *= max(0.0, hit_normal_ws_dot.w) / max(1e-4, hit_normal_ws_dot.w);
    }*/

#if 1
    /*if (!use_resampling) {
        reservoir.w_sum = (calculate_luma(result.out_value));
        reservoir.w_sel = reservoir.w_sum;
        reservoir.W = 1;
        reservoir.M = 1;
    }*/

    /*if (result.out_value.r > prev_irrad.r * 1.5 + 0.1) {
        result.out_value.b = 1000;
    }*/
    //result.out_value = min(result.out_value, prev_irrad * 1.5 + 0.1);

    irradiance_out_tex[px] = float4(irradiance_sel, pdf_sel);
    ray_orig_output_tex[px] = float4(ray_orig_sel, length(ray_hit_sel - refl_ray_origin));
    //irradiance_out_tex[px] = float4(result.out_value, dot(gbuffer.normal, outgoing_ray.Direction));
    hit_normal_output_tex[px] = hit_normal_ws_dot;
    ray_output_tex[px] = float4(ray_hit_sel/* - get_eye_position()*/, ratio_estimator_factor);
    reservoir_out_tex[px] = reservoir.as_raw();
#endif
}
