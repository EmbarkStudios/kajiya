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
#include "restir_settings.hlsl"

// Should be 1, but rarely matters for the diffuse bounce, so might as well save a few cycles.
#define USE_SOFT_SHADOWS 0

#define USE_SURFEL_GI 1

#define USE_TEMPORAL_JITTER 1
#define USE_SHORT_RAYS_ONLY 0
#define SHORT_RAY_SIZE_VOXEL_CELLS 4.0
#define ROUGHNESS_BIAS 0.5
#define SUPPRESS_GI_FOR_NEAR_HITS 1
#define USE_SCREEN_GI_REPROJECTION 0

#define USE_EMISSIVE 1
#define USE_LIGHTS 1

[[vk::binding(0)]] Texture2D<float3> half_view_normal_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> candidate_irradiance_tex;
[[vk::binding(3)]] Texture2D<float4> candidate_hit_tex;
DEFINE_BLUE_NOISE_SAMPLER_BINDINGS(4, 5, 6)
[[vk::binding(7)]] Texture2D<float4> irradiance_history_tex;
[[vk::binding(8)]] Texture2D<float4> ray_history_tex;
[[vk::binding(9)]] Texture2D<float4> reservoir_history_tex;
[[vk::binding(10)]] Texture2D<float4> reprojection_tex;
[[vk::binding(11)]] Texture2D<float4> hit_normal_history_tex;
[[vk::binding(12)]] Texture2D<float4> candidate_history_tex;
[[vk::binding(13)]] RWTexture2D<float4> irradiance_out_tex;
[[vk::binding(14)]] RWTexture2D<float4> ray_out_tex;
[[vk::binding(15)]] RWTexture2D<float4> hit_normal_tex;
[[vk::binding(16)]] RWTexture2D<float4> reservoir_out_tex;
[[vk::binding(17)]] RWTexture2D<float4> candidate_out_tex;
[[vk::binding(18)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

#include "candidate_ray_dir.hlsl"

static const float SKY_DIST = 1e4;

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

struct TraceResult {
    float3 out_value;
    float3 hit_normal_ws;
    float hit_t;
    float inv_pdf;
};

TraceResult do_the_thing(uint2 px, inout uint rng, RayDesc outgoing_ray, float3 primary_hit_normal) {
    const float4 candidate_irradiance_inv_pdf = candidate_irradiance_tex[px];
    TraceResult result;
    result.out_value = candidate_irradiance_inv_pdf.rgb;
    result.inv_pdf = candidate_irradiance_inv_pdf.a;
    float4 hit = candidate_hit_tex[px];
    result.hit_t = hit.w;
    result.hit_normal_ws = hit.xyz;
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
        hit_normal_tex[px] = 0.0.xxxx;
        reservoir_out_tex[px] = 0.0.xxxx;
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    //float4 gbuffer_packed = gbuffer_tex[hi_px];
    //GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    const float3 normal_vs = half_view_normal_tex[px];
    const float3 normal_ws = direction_view_to_world(normal_vs);

    const float3x3 tangent_to_world = build_orthonormal_basis(normal_ws);
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

    // TODO: use
    float3 light_radiance = 0.0.xxx;

    // HACK; should be in dedicated passes
    /*if (USE_LIGHTS) {
        float2 urand = blue_noise_for_pixel(px, frame_constants.frame_index + 100).xy;

        for (uint light_idx = 0; light_idx < frame_constants.triangle_light_count; light_idx += 1) {
            TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
            LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
            const float3 shadow_ray_origin = view_ray_context.ray_hit_ws();
            const float3 to_light_ws = light_sample.pos - shadow_ray_origin;
            const float dist_to_light2 = dot(to_light_ws, to_light_ws);
            const float3 to_light_norm_ws = to_light_ws * rsqrt(dist_to_light2);

            const float to_psa_metric =
                max(0.0, dot(to_light_norm_ws, normal_ws))
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
    }*/

    float3 outgoing_dir = rtdgi_candidate_ray_dir(px, tangent_to_world);

    float p_q_sel = 0;
    uint2 src_px_sel = px;
    float3 irradiance_sel = 0;
    float3 ray_hit_sel = 1;
    float3 hit_normal_sel = 1;
    uint sel_valid_sample_idx = 0;

    Reservoir1spp reservoir = Reservoir1spp::create();
    const uint reservoir_payload = px.x | (px.y << 16);

    reservoir.payload = reservoir_payload;

    {
        RayDesc outgoing_ray;
        outgoing_ray.Direction = outgoing_dir;
        outgoing_ray.Origin = refl_ray_origin;
        outgoing_ray.TMin = 0;
        outgoing_ray.TMax = SKY_DIST;

        TraceResult result = do_the_thing(px, rng, outgoing_ray, normal_ws);

        const float p_q = p_q_sel =
            max(1e-3, calculate_luma(result.out_value))
            #if !DIFFUSE_GI_BRDF_SAMPLING
                * max(0, dot(outgoing_dir, normal_ws))
            #endif
            ;

        const float inv_pdf_q = result.inv_pdf;

        irradiance_sel = result.out_value;
        ray_hit_sel = outgoing_ray.Origin + outgoing_ray.Direction * result.hit_t;
        hit_normal_sel = result.hit_normal_ws;

        reservoir.payload = reservoir_payload;
        reservoir.w_sum = p_q * inv_pdf_q;
        reservoir.M = 1;
        reservoir.W = inv_pdf_q;

        float rl = lerp(candidate_history_tex[px].y, sqrt(result.hit_t), 0.05);
        candidate_out_tex[px] = float4(sqrt(result.hit_t), rl, 0, 0);
    }

    const float4 reproj = reprojection_tex[hi_px];

    //const bool use_resampling = false;
    const bool use_resampling = DIFFUSE_GI_USE_RESTIR;

    if (use_resampling && reproj.z != 0) {
        float M_sum = reservoir.M;

        // Can't use linear interpolation, but we can interpolate stochastically instead
        //const float2 reproj_rand_offset = float2(uint_to_u01_float(hash1_mut(rng)), uint_to_u01_float(hash1_mut(rng))) - 0.5;
        // Or not at all.
        const float2 reproj_rand_offset = 0.0;
        int2 reproj_px = floor(px + gbuffer_tex_size.xy * reproj.xy / 2 + reproj_rand_offset + 0.5);

        uint valid_sample_count = 0;
        const float ang_offset = uint_to_u01_float(hash1_mut(rng)) * M_PI * 2;
        for (uint sample_i = 0; sample_i < 1; ++sample_i) {
            const int2 rpx = reproj_px;
            const uint2 rpx_hi = rpx * 2 + hi_px_offset;

            Reservoir1spp r = Reservoir1spp::from_raw(reservoir_history_tex[rpx]);

            const float3 sample_normal_vs = half_view_normal_tex[rpx];

            if (sample_i > 0 && dot(sample_normal_vs, normal_vs) < 0.9) {
                continue;
            }

            const float2 sample_uv = get_uv(rpx_hi, gbuffer_tex_size);
            const float sample_depth = depth_tex[rpx_hi];
            if (sample_i > 0 && 0 == sample_depth) {
                continue;
            }

            const ViewRayContext sample_ray_ctx = ViewRayContext::from_uv_and_depth(sample_uv, sample_depth);
            const float3 sample_origin_ws = sample_ray_ctx.biased_secondary_ray_origin_ws();

            const float4 prev_hit_ws_and_dist = ray_history_tex[rpx];
            const float3 prev_hit_ws = prev_hit_ws_and_dist.xyz;
            const float prev_dist = prev_hit_ws_and_dist.w;
            //const float prev_dist = length(prev_hit_ws - sample_origin_ws);

            /*if (sample_i > 0 && !(prev_dist > 1e-4)) {
                continue;
            }*/

            const float3 sample_dir_unnorm = prev_hit_ws - refl_ray_origin;
            const float sample_dist = length(sample_dir_unnorm);
            const float3 sample_dir = normalize(sample_dir_unnorm);

            if (sample_i > 0 && dot(sample_dir, normal_ws) < 1e-3) {
                continue;
            }
            
            const float4 prev_irrad = irradiance_history_tex[rpx];

            //if (prev_irrad.r > prev_irrad.b) {continue;}

            // TODO: need the previous normal (last frame)
            //const float4 prev_hit_normal_ws_dot = hit_normal_tex[rpx];

            // From the ReSTIR paper:
            // With temporal reuse, the number of candidates M contributing to the
            // pixel can in theory grow unbounded, as each frame always combines
            // its reservoir with the previous frame’s. This causes (potentially stale)
            // temporal samples to be weighted disproportionately high during
            // resampling. To fix this, we simply clamp the previous frame’s M
            // to at most 20× of the current frame’s reservoir’s M

            r.M = min(r.M, RESTIR_TEMPORAL_M_CLAMP);

            float p_q = 1;
            p_q *= max(1e-3, calculate_luma(prev_irrad.rgb));
            #if !DIFFUSE_GI_BRDF_SAMPLING
                p_q *= max(0, dot(sample_dir, normal_ws));
            #endif

            float visibility = 1;

            const float4 prev_hit_normal_ws_dot = hit_normal_history_tex[rpx];

            float jacobian = 1;

            // Note: needed for sample 0 due to temporal jitter.
            //if (sample_i > 0)
            {
                // Distance falloff. Needed to avoid leaks.
                jacobian *= clamp(prev_dist, 1e-3, 1e3) / clamp(sample_dist, 1e-3, 1e3);
                jacobian *= jacobian;

                // N of hit dot -L. Needed to avoid leaks.
                jacobian *= max(0.0, -dot(prev_hit_normal_ws_dot.xyz, sample_dir)) / max(1e-4, prev_hit_normal_ws_dot.w);

                // Note: causes flicker due to normal differences between frames (TAA, half-res downsample jitter).
                // Might be better to apply at the end, in spatial resolve. When used with the bias,
                // causes severe darkening instead (on bumpy normal mapped surfaces).
                //
                // N dot L. Useful for normal maps, micro detail.
                // The min(const, _) should not be here, but it prevents fireflies and brightening of edges
                // when we don't use a harsh normal cutoff to exchange reservoirs with.
                //jacobian *= min(1, max(0.0, prev_irrad.a) / dot(sample_dir, normal_ws));
                //jacobian *= max(0.0, prev_irrad.a) / dot(sample_dir, normal_ws);
                // TODO: find best pixel to reproject to
            }

            M_sum += r.M;
            if (reservoir.update(p_q * r.W * r.M * jacobian * visibility, reservoir_payload, rng)) {
                outgoing_dir = sample_dir;
                p_q_sel = p_q;
                jacobian = jacobian;
                src_px_sel = rpx;
                irradiance_sel = prev_irrad.rgb;
                ray_hit_sel = prev_hit_ws;
                hit_normal_sel = prev_hit_normal_ws_dot.xyz;
                sel_valid_sample_idx = valid_sample_count;
            }
        }

        valid_sample_count = max(valid_sample_count, 1);

        reservoir.M = M_sum;
        reservoir.W = (1.0 / max(1e-5, p_q_sel)) * (reservoir.w_sum / reservoir.M);
    }

    RayDesc outgoing_ray;
    outgoing_ray.Direction = outgoing_dir;
    outgoing_ray.Origin = refl_ray_origin;
    outgoing_ray.TMin = 0;

    //TraceResult result = do_the_thing(px, rng, outgoing_ray, gbuffer);

    const float4 hit_normal_ws_dot = float4(hit_normal_sel, -dot(hit_normal_sel, outgoing_ray.Direction));

    /*if (any(src_px_sel != px)) {
        const uint2 spx = src_px_sel;
        const float4 prev_hit_normal_ws_dot = hit_normal_history_tex[spx];
        jacobian *= max(0.0, hit_normal_ws_dot.w) / max(1e-4, prev_hit_normal_ws_dot.w);
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

    irradiance_out_tex[px] = float4(irradiance_sel, dot(normal_ws, outgoing_ray.Direction));
    //irradiance_out_tex[px] = float4(result.out_value, dot(gbuffer.normal, outgoing_ray.Direction));
    hit_normal_tex[px] = hit_normal_ws_dot;
    ray_out_tex[px] = float4(ray_hit_sel, length(ray_hit_sel - refl_ray_origin));
    reservoir_out_tex[px] = reservoir.as_raw();
#endif
}
