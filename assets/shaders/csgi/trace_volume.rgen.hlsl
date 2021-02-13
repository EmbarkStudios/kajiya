#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/sun.hlsl"

#include "../inc/atmosphere.hlsl"
static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

#include "common.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0)]] RWTexture3D<float4> out0_tex;
[[vk::binding(1)]] cbuffer _ {
    float4 SLICE_DIRS[16];
}

static const float SKY_DIST = 1e5;

float3 vx_to_pos(float3 vx, float3x3 slice_rot) {
    return mul(slice_rot, vx - (GI_VOLUME_DIMS - 1.0) / 2.0) * GI_VOLUME_SCALE + GI_VOLUME_CENTER;
}

[shader("raygeneration")]
void main() {
    uint3 px = DispatchRaysIndex().xyz;
    const uint grid_idx = px.x / GI_VOLUME_DIMS;
    px.x %= GI_VOLUME_DIMS;

    const float3x3 slice_rot = build_orthonormal_basis(SLICE_DIRS[grid_idx].xyz);
    const float3 slice_dir = mul(slice_rot, float3(0, 0, -1));    
    const float3 trace_origin = vx_to_pos(px + float3(0, 0, 0.5), slice_rot);

    RayDesc outgoing_ray;
    outgoing_ray.Direction = vx_to_pos(px + float3(0, 0, -1), slice_rot) - trace_origin;
    outgoing_ray.Origin = trace_origin;
    outgoing_ray.TMin = 0;
    outgoing_ray.TMax = 1.0;

    float3 radiance = 0.0.xxx;

    /*const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray);
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
                        SKY_DIST
                ));

            GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
            const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
            const float3 wi = mul(to_light_norm, shading_basis);
            float3 wo = normalize(mul(-outgoing_ray.Direction, shading_basis));

            LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

            const float3 brdf_value = brdf.evaluate(wo, wi);
            const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
            total_radiance += brdf_value * light_radiance;

            //total_radiance *= saturate(dot(-gbuffer.normal, normalize(outgoing_ray.Direction)));
        }

        radiance += total_radiance;
    }
    else*/
    {
        const float spread0 = 1.0;
        const float spread1 = 0.7;

        float3 scatter = 0.0;

        static const uint DIR_COUNT = 9;
        static const int3 dirs[9] = {
            int3(0, 0, -1),

            int3(1, 0, -1),
            int3(-1, 0, -1),
            int3(0, 1, -1),
            int3(0, -1, -1),

            int3(1, 1, -1),
            int3(-1, 1, -1),
            int3(1, -1, -1),
            int3(-1, -1, -1)
        };
        static const float weights[9] = {
            1,
            spread0, spread0, spread0, spread0,
            spread1, spread1, spread1, spread1
        };

        float total_wt = 0;

        [unroll]
        for (uint dir_i = 0; dir_i < DIR_COUNT; ++dir_i)
        {
            //const uint dir_i = 0;
            total_wt += weights[dir_i];

            float3 r_dir = mul(slice_rot, float3(dirs[dir_i]));
            float3 neighbor_pos = vx_to_pos(float3(px) + dirs[dir_i] * 1.5, slice_rot);    // 1.5 to hit corner/back wall

            RayDesc outgoing_ray = new_ray(
                trace_origin,
                neighbor_pos - trace_origin,
                0,
                1.0
            );

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
                                SKY_DIST
                        ));

                    GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();

                    const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
                    const float3 wi = mul(to_light_norm, shading_basis);
                    float3 wo = normalize(mul(-outgoing_ray.Direction, shading_basis));

                    LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);
                    const float3 brdf_value = brdf.evaluate(wo, wi);
                    const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
                    total_radiance += brdf_value * light_radiance;




                    total_radiance *= saturate(0.0 + dot(-gbuffer.normal, normalize(outgoing_ray.Direction)));
                }

                //if (dir_i > 0)
                scatter += total_radiance * weights[dir_i];
            } else {
                int3 src_px = int3(px) + dirs[dir_i];
                if (src_px.x >= 0 && src_px.x < GI_VOLUME_DIMS) {
                    scatter += out0_tex[src_px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx].rgb * weights[dir_i];
                }
            }
            
            /*if (!rt_is_shadowed(
                acceleration_structure,
                new_ray(
                    trace_origin,
                    neighbor_pos - trace_origin,
                    0,
                    1.0
            ))) {
                scatter += out0_tex[int3(px) + dirs[dir_i]].xyz * weights[dir_i];
            }*/
        }

        radiance += scatter.xyz / total_wt;
    }

    out0_tex[px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx] = float4(radiance, 1);
}
