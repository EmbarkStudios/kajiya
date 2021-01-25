#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/rt.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;


[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    float depth = depth_tex[px];

    if (0.0 == depth) {
        output_tex[px] = 0.0.xxxx;
        return;
    }

    const float2 uv = (px + 0.5) / DispatchRaysDimensions().xy;
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    float4 gbuffer_packed = gbuffer_tex[px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();

    const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);

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

    uint seed = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);
    const float2 urand = float2(
        uint_to_u01_float(hash1_mut(seed)),
        uint_to_u01_float(hash1_mut(seed)),
    );
    BrdfSample brdf_sample = specular_brdf.sample(wo, urand);

    if (brdf_sample.is_valid()) {
        RayDesc outgoing_ray;
        outgoing_ray.Origin = primary_hit.position;
        outgoing_ray.Direction = mul(shading_basis, brdf_sample.wi);
        outgoing_ray.TMin = 1e-4;
        outgoing_ray.TMax = FLT_MAX;

        const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray);
        if (primary_hit.is_hit) {
            GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
            output_tex[px] = float4(gbuffer.albedo, 1);

    /*const float3 to_light_norm = SUN_DIRECTION;
    const bool is_shadowed =
        (INDIRECT_ONLY && path_length == 0) ||
        path_length+1 >= MAX_PATH_LENGTH ||
        rt_is_shadowed(
            acceleration_structure,
            new_ray(
                primary_hit.position,
                to_light_norm,
                1e-4,
                FLT_MAX
        ));*/
        } else {
            output_tex[px] = 0.0.xxxx;
        }
    } else {
        output_tex[px] = 0.0.xxxx;
    }
}
