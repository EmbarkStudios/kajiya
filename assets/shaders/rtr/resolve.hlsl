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
[[vk::binding(4)]] RWTexture2D<float4> output_tex;
[[vk::binding(5)]] cbuffer _ {
    float4 output_tex_size;
    int4 spatial_resolve_offsets[16 * 4 * 8];
};

#define USE_APPROX_BRDF 1

float inverse_lerp(float minv, float maxv, float v) {
    return (v - minv) / (maxv - minv);
}

float approx_fresnel(float3 wo, float3 wi) {
   float3 h_unnorm = wo + wi;
   return exp2(-1.0 -8.65617024533378 * dot(wi, h_unnorm));
}

bool is_wave_alive(uint mask, uint idx) {
    return (mask & (1 << idx)) != 0;
}

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    const float2 uv = get_uv(px, output_tex_size);
    const float depth = depth_tex[px];

    if (0.0 == depth) {
        output_tex[px] = 0.0.xxxx;
        return;
    }

    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    const float4 gbuffer_packed = gbuffer_tex[px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();

    const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
    float3 wo = mul(-normalize(view_ray_context.ray_dir_ws()), shading_basis);

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

    const uint px_idx_in_quad = (px.x & 1) | (px.y & 1) * 2;
    const uint sample_count = 16;

    const float a2 = gbuffer.roughness * gbuffer.roughness;

    // Project lobe footprint onto the reflector, and find the desired convolution size
    const float surf_to_hit_dist = abs(hit0_tex[px / 2].w);
    const float eye_to_surf_dist = -view_ray_context.ray_hit_vs().z;
    const float filter_spread = surf_to_hit_dist / (surf_to_hit_dist + eye_to_surf_dist);

    // Reduce size estimate variance by shrinking it across neighbors
    float filter_size = filter_spread * gbuffer.roughness * 16;

    const uint wave_mask = WaveActiveBallot(true).x;
    if (is_wave_alive(wave_mask, WaveGetLaneIndex() ^ 2)) {
        filter_size = min(filter_size, WaveReadLaneAt(filter_size, WaveGetLaneIndex() ^ 2));
    }
    if (is_wave_alive(wave_mask, WaveGetLaneIndex() ^ 16)) {
        filter_size = min(filter_size, WaveReadLaneAt(filter_size, WaveGetLaneIndex() ^ 16));
    }

    // Choose one of a few pre-baked sample sets based on the footprint
    const uint filter_idx = uint(clamp(filter_size * 8, 0, 7));
    //output_tex[px] = float4((filter_idx / 7.0).xxx, 0);
    //return;

    float4 contrib_sum = 0.0;

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        const int2 sample_px = px / 2 + spatial_resolve_offsets[(px_idx_in_quad * 16 + sample_i) + 64 * filter_idx].xy;

        float sample_depth = depth_tex[sample_px * 2];

        float4 packed0 = hit0_tex[sample_px];
        if (packed0.w > 0) {
            float4 packed1 = hit1_tex[sample_px];

            // TODO: use packed gb
            GbufferData sample_gbuffer =
                GbufferDataPacked::from_uint4(asuint(gbuffer_tex[sample_px * 2])).unpack();

            const float3 wi = mul(packed1.xyz, shading_basis);
            if (wi.z > 1e-5) {

    #if !USE_APPROX_BRDF
                BrdfValue spec = specular_brdf.evaluate(wo, wi);
                float spec_weight = spec.value().x;
    #else
                float spec_weight;
                {
                    const float3 m = normalize(wo + wi);
                    const float cos_theta = m.z;
                    const float pdf_h_over_cos_theta = SpecularBrdf::ggx_ndf(a2, cos_theta);
                    //const float f = lerp(f0_grey, 1.0, approx_fresnel(wo, wi));
                    const float f = eval_fresnel_schlick(f0_grey, 1.0, dot(m, wi)).x;

                    spec_weight = f * pdf_h_over_cos_theta / max(1e-5, wo.z + wi.z);
                }
    #endif

                float rejection_bias =
                    saturate(inverse_lerp(0.2, 0.6, dot(gbuffer.normal, sample_gbuffer.normal)));

                rejection_bias *= exp2(-10.0 * abs(1.0 / sample_depth - 1.0 / depth) * depth);

                // TODO: approx shadowing

                const float contrib_wt = spec_weight * rejection_bias / packed1.w;
                contrib_sum += float4(packed0.rgb, 1) * contrib_wt;
            }
        }
    }    

    output_tex[px] = contrib_sum / max(1e-5, contrib_sum.w);
}
