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
};

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
    specular_brdf.albedo = 1.0;
    specular_brdf.roughness = gbuffer.roughness;

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

    float4 contrib_sum = 0.0;
    for (uint sample_i = 0; sample_i < 9; ++sample_i) {
        const int2 sample_px = px / 2 + sample_offsets[sample_i];

        float4 packed0 = hit0_tex[sample_px];
        if (packed0.w != 0) {
            float4 packed1 = hit1_tex[sample_px];

            const float3 wi = mul(packed1.xyz, shading_basis);
            BrdfValue spec = specular_brdf.evaluate(wo, wi);

            contrib_sum +=
                float4(packed0.rgb, 1)
                * spec.value().x
                / clamp(packed1.w, 1e-5, 1e5)
                //* max(0.0, wi.z)
                * step(0.0, wi.z)
                ;
        }
    }    

    output_tex[px] = contrib_sum / max(1e-5, contrib_sum.w);
}
