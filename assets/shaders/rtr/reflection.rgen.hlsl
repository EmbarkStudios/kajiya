#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/sun.hlsl"

#include "../inc/atmosphere.hlsl"
static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] RWTexture2D<float4> out0_tex;
[[vk::binding(3)]] RWTexture2D<float4> out1_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 gbuffer_tex_size;
};


[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    const uint2 hi_px = px * 2;
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        out0_tex[px] = 0.0.xxxx;
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    float4 gbuffer_packed = gbuffer_tex[hi_px];
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
    specular_brdf.albedo = lerp(0.04, gbuffer.albedo, gbuffer.metalness);
    specular_brdf.roughness = gbuffer.roughness;

    const float sampling_bias = 0.3;

    uint seed = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);
    const float2 urand = float2(
        lerp(uint_to_u01_float(hash1_mut(seed)), 0.0, sampling_bias),
        uint_to_u01_float(hash1_mut(seed))
    );
    BrdfSample brdf_sample = specular_brdf.sample(wo, urand);

    if (brdf_sample.is_valid()) {
        RayDesc outgoing_ray;
        outgoing_ray.Origin = view_ray_context.ray_hit_ws();
        outgoing_ray.Direction = mul(shading_basis, brdf_sample.wi);
        outgoing_ray.TMin = 1e-3;
        outgoing_ray.TMax = FLT_MAX;

        out1_tex[px] = float4(outgoing_ray.Direction, brdf_sample.pdf);

        const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray);
        if (primary_hit.is_hit) {
            float3 total_radiance = 0.0.xxx;
            {
                const float3 to_light_norm = SUN_DIRECTION;
                const bool is_shadowed =
                    rt_is_shadowed(
                        acceleration_structure,
                        new_ray(
                            primary_hit.position,
                            to_light_norm,
                            1e-4,
                            FLT_MAX
                    ));

                GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
                const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
                const float3 wi = mul(to_light_norm, shading_basis);
                float3 wo = mul(-outgoing_ray.Direction, shading_basis);

                const LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);
                const float3 brdf_value = brdf.evaluate(wo, wi);
                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
                total_radiance += brdf_value * light_radiance;
            }

            out0_tex[px] = float4(total_radiance, 1);
        } else {
            out0_tex[px] = float4(atmosphere_default(outgoing_ray.Direction, SUN_DIRECTION), 1);
        }
    } else {
        out0_tex[px] = 0.0.xxxx;
    }
}
