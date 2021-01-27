#include "../inc/color.hlsl"
#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"

#include "spatial_samples.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> hit0_tex;
[[vk::binding(3)]] Texture2D<float4> hit1_tex;
[[vk::binding(4)]] RWTexture2D<float4> output_tex;
[[vk::binding(5)]] cbuffer _ {
    float4 output_tex_size;
};

#define USE_APPROX_BRDF 1

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

#if !USE_APPROX_BRDF
    SpecularBrdf specular_brdf;
    specular_brdf.albedo = 1.0;
    specular_brdf.roughness = gbuffer.roughness;
#endif

    static const int2 sample_offsets[9] = {
        int2(0, 0),

        int2(-2, 0),
        int2(2, 0),
        int2(0, -2),
        int2(0, 2),

        int2(-3, -3),
        int2(-3, 3),
        int2(3, -3),
        int2(3, 3),
    };

    const uint px_idx_in_quad = (px.x & 1) | (px.y & 1) * 2;
    const uint sample_set_idx = 0;//frame_constants.frame_index & 3;
    const uint sample_count = 16;

    const float a2 = gbuffer.roughness * gbuffer.roughness;

    // TODO: roughness-and-distance scaling sample sets
    const int sample_offset_scale = gbuffer.roughness < 0.3 ? 1 : 2;

    float4 contrib_sum = 0.0;
    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        //const int2 sample_px = px / 2 + sample_offsets[sample_i];
        const int2 sample_px = px / 2 + sample_offset_scale * spatial_filter_sample_offsets[sample_set_idx][px_idx_in_quad][sample_i];

        float4 packed0 = hit0_tex[sample_px];
        if (packed0.w != 0) {
            float4 packed1 = hit1_tex[sample_px];

            // TODO: use packed gb
            GbufferData sample_gbuffer =
                GbufferDataPacked::from_uint4(asuint(gbuffer_tex[sample_px * 2])).unpack();

            const float3 wi = mul(packed1.xyz, shading_basis);

#if !USE_APPROX_BRDF            
            BrdfValue spec = specular_brdf.evaluate(wo, wi);
            float spec_weight = spec.value().x;
#else
            float spec_weight;
            {
                const float3 m = normalize(wo + wi);
                const float cos_theta = m.z;
                const float pdf_h_denom_sqrt = 1.0 + (-1.0 + a2) * cos_theta * cos_theta;
                const float pdf_h_over_cos_theta = SpecularBrdf::ggx_ndf(a2, cos_theta);

                spec_weight = pdf_h_over_cos_theta / max(1e-5, wo.z + wi.z);
            }
#endif

            const float rejection_bias =
                max(0.0, 3.0 * dot(gbuffer.normal, sample_gbuffer.normal));

            // TODO: approx shadowing

            const float contrib_wt = spec_weight * rejection_bias / packed1.w * step(0.0, wi.z);
            contrib_sum += float4(packed0.rgb, 1) * contrib_wt;
        }
    }    

    output_tex[px] = contrib_sum / max(1e-5, contrib_sum.w);
}
