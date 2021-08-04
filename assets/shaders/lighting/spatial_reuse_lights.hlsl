#include "../inc/color.hlsl"
#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> hit0_tex;
[[vk::binding(3)]] Texture2D<float4> hit1_tex;
[[vk::binding(4)]] Texture2D<float4> hit2_tex;
[[vk::binding(5)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(6)]] Texture2D<float> half_depth_tex;
[[vk::binding(7)]] RWTexture2D<float4> output_tex;
[[vk::binding(8)]] cbuffer _ {
    float4 output_tex_size;
    int4 spatial_resolve_offsets[16 * 4 * 8];
};

#define SHUFFLE_SUBPIXELS 1
#define BORROW_SAMPLES 1
#define USE_APPROXIMATE_SAMPLE_SHADOWING 1

// See RTR_NEIGHBOR_RAY_ORIGIN_CENTER_BIAS
#define NEIGHBOR_RAY_ORIGIN_CENTER_BIAS 0.5

#define RENDER_INTO_RTR 1

float inverse_lerp(float minv, float maxv, float v) {
    return (v - minv) / (maxv - minv);
}

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID) {
    uint2 orig_px = px;

    const float2 uv = get_uv(px, output_tex_size);
    const float depth = depth_tex[px];

    if (0.0 == depth) {
        #if !RENDER_INTO_RTR
            output_tex[px] = 0.0.xxxx;
        #endif

        return;
    }

    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    const float4 gbuffer_packed = gbuffer_tex[px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    
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

    float3 energy_preservation_mult = 1;
    SpecularBrdf specular_brdf;
    {
        LayeredBrdf layered_brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);
        specular_brdf = layered_brdf.specular_brdf;

        // TODO: integrate into te specular BRDF so it doesn't need explicit handling everywhere
        energy_preservation_mult = layered_brdf.energy_preservation.preintegrated_reflection_mult;
    }

    // Index used to calculate a sample set disjoint for all four pixels in the quad
    // Offsetting by frame index reduces small structured artifacts
    const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + (SHUFFLE_SUBPIXELS ? 1 : 0) * frame_constants.frame_index) & 3;

    const uint sample_count = BORROW_SAMPLES ? 8 : 1;
    const uint filter_idx = 3;

    float4 contrib_accum = 0.0;

    const float3 normal_vs = direction_world_to_view(gbuffer.normal);

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        // TODO: precalculate temporal variants
        const int2 sample_offset = spatial_resolve_offsets[(px_idx_in_quad * 16 + sample_i) + 64 * filter_idx].xy;
        const int2 sample_px = px / 2 + sample_offset;
        const float sample_depth = half_depth_tex[sample_px];

        const float4 packed0 = hit0_tex[sample_px];

        if (packed0.w != 0 && sample_depth != 0) {
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

            const float3 center_to_hit_vs = packed1.xyz - lerp(view_ray_context.ray_hit_vs(), sample_origin_vs, NEIGHBOR_RAY_ORIGIN_CENTER_BIAS);
            const float3 sample_hit_vs = center_to_hit_vs + view_ray_context.ray_hit_vs();

            const float3 wi = normalize(mul(direction_view_to_world(center_to_hit_vs), tangent_to_world));
            const float3 sample_normal_vs = half_view_normal_tex[sample_px].xyz;

            float rejection_bias = 1;
            // TODO: brdf-driven blend
            rejection_bias *= saturate(inverse_lerp(0.9, 0.999, dot(normal_vs, sample_normal_vs)));
            rejection_bias *= exp2(-10.0 * abs(depth / sample_depth - 1.0));

            const float sample_hit_to_center_vis = saturate(dot(sample_hit_normal_vs, -normalize(center_to_hit_vs)));

            {
        		const float3 surface_offset = sample_origin_vs - view_ray_context.ray_hit_vs();
                const float fraction_of_normal_direction_as_offset = dot(surface_offset, normal_vs) / length(surface_offset);

                #if USE_APPROXIMATE_SAMPLE_SHADOWING
                    // Note: the `wi.z > 0` is needed to avoid excessive bias when light samples are behind the normal plane, and would
                    // be rejected by the bias, thus skewing sample counting.
                    //if (wi.z > 0 && dot(center_to_hit_vs, normal_vs) * 0.2 / length(center_to_hit_vs) < dot(surface_offset, normal_vs) / length(surface_offset)) {
                    if (wi.z > 0 && wi.z * 0.2 < fraction_of_normal_direction_as_offset) {
            			rejection_bias *= sample_i == 0 ? 1 : 0;
            		}
                #endif
            }

            const BrdfValue spec = specular_brdf.evaluate(wo, wi);

            const float center_to_hit_dist2 = dot(center_to_hit_vs, center_to_hit_vs);

            // Convert to projected solid angle
            const float to_psa_metric =
                max(0, wi.z) * max(0, dot(sample_hit_normal_vs, -normalize(center_to_hit_vs)))
                / center_to_hit_dist2;
            neighbor_sampling_pdf /= to_psa_metric;

            // Note: should indeed be step(0, wi.z) since the cosine factor is part
            // of the measure conversion from area to projected solid angle.
            const float3 contrib_rgb = packed0.rgb * spec.value * energy_preservation_mult * step(0.0, wi.z) * (neighbor_sampling_pdf > 0 ? (1 / neighbor_sampling_pdf) : 0);
            const float contrib_wt = rejection_bias;

            contrib_accum += float4(contrib_rgb, 1) * contrib_wt;

            float luma = calculate_luma(packed0.rgb);
        }
    }

    const float contrib_norm_factor = max(1e-8, contrib_accum.w);
    contrib_accum.rgb /= contrib_norm_factor;

    float3 out_color = contrib_accum.rgb;

#if RENDER_INTO_RTR
    output_tex[orig_px].rgb += out_color;
#else
    output_tex[orig_px] = float4(out_color, 0);
#endif
}
