#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/sun.hlsl"

#include "../inc/atmosphere.hlsl"
static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

#include "common.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0)]] RWTexture3D<float4> cascade0_tex;
[[vk::binding(1)]] RWTexture3D<float4> integr_tex;
[[vk::binding(2)]] cbuffer _ {
    float4 SLICE_DIRS[16];
}

static const float SKY_DIST = 1e5;

float3 vx_to_pos(float3 vx, float3x3 slice_rot) {
    return mul(slice_rot, vx - (GI_VOLUME_DIMS - 1.0) / 2.0) * GI_VOXEL_SIZE + gi_volume_center(slice_rot);
}

// HACK; broken
#define CSGI_LOOKUP_NEAREST_ONLY
#define alt_cascade0_tex cascade0_tex
#include "lookup.hlsl"

static const bool USE_MULTIBOUNCE = true;


[shader("raygeneration")]
void main() {
    uint3 px = DispatchRaysIndex().xyz;
    const uint grid_idx = px.x / GI_VOLUME_DIMS;
    px.x %= GI_VOLUME_DIMS;

    const float3x3 slice_rot = build_orthonormal_basis(SLICE_DIRS[grid_idx].xyz);
    const float3 slice_dir = mul(slice_rot, float3(0, 0, -1));    

    const float spread0 = 1.0;
    //const float spread1 = 0.7;
    const float spread1 = 1.0;

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

    //uint rng = hash_combine2(hash3(px), frame_constants.frame_index);
    //uint rng = hash1(frame_constants.frame_index);
    uint rng = hash2(uint2(frame_constants.frame_index, grid_idx));
    //uint rng = hash2(uint2(px.y, frame_constants.frame_index));

    //for (uint dir_i = 0; dir_i < DIR_COUNT; ++dir_i)
    //uint dir_i = rng % DIR_COUNT;
    uint dir_i = frame_constants.frame_index % DIR_COUNT;
    //uint dir_i = (frame_constants.frame_index * 5) % DIR_COUNT;
    {
        //uint rng = hash2(uint2(dir_i, frame_constants.frame_index));

        //const uint dir_i = 0;
        total_wt += weights[dir_i];

        #if 0
            const float offset_x = uint_to_u01_float(hash1_mut(rng)) - 0.5;
            const float offset_y = uint_to_u01_float(hash1_mut(rng)) - 0.5;
            const float blend_factor = 0.5;
        #else
            const float offset_x = 0.0;
            const float offset_y = 0.0;
            const float blend_factor = 1.0;
        #endif
        
        const float3 trace_origin = vx_to_pos(px + float3(offset_x, offset_y, 0.5), slice_rot);
        const float3 neighbor_pos = vx_to_pos(float3(px) + dirs[dir_i] + float3(offset_x, offset_y, 0.5), slice_rot);

        RayDesc outgoing_ray = new_ray(
            trace_origin,
            neighbor_pos - trace_origin,
            0,
            1.0
        );

        const int3 preintegr_px = px * int3(1, DIR_COUNT, 1) + int3(GI_VOLUME_DIMS * grid_idx, dir_i, 0);

        const GbufferPathVertex primary_hit = rt_trace_gbuffer_nocull(acceleration_structure, outgoing_ray);
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
                
                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;

                // Compensate for lost specular (TODO)
                const float3 bounce_albedo = lerp(gbuffer.albedo, 1.0.xxx, 0.04);
                //const float3 bounce_albedo = gbuffer.albedo;

#if 0
                const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
                const float3 wi = mul(to_light_norm, shading_basis);
                float3 wo = normalize(mul(-outgoing_ray.Direction, shading_basis));

                LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);
                const float3 brdf_value = brdf.evaluate(wo, wi);
                total_radiance += brdf_value * light_radiance;
#else
                total_radiance +=
                    light_radiance * bounce_albedo * max(0.0, dot(gbuffer.normal, to_light_norm)) / M_PI;
#endif

                if (USE_MULTIBOUNCE) {
                    CsgiLookupParams gi_lookup_params;
                    gi_lookup_params.use_grid_linear_fetch = false;
                    gi_lookup_params.use_pretrace = false;
                    gi_lookup_params.debug_slice_idx = -1;

                    total_radiance += lookup_csgi(primary_hit.position, gbuffer.normal, gi_lookup_params) * bounce_albedo;
                }

                // Remove contributions where the normal is facing away from
                // the cone direction. This can happen e.g. on rays travelling near floors,
                // where the neighbor rays hit the floors, but contribution should be zero.
                total_radiance *= smoothstep(0.0, 0.1, dot(slice_dir, -gbuffer.normal));
                //total_radiance *= step(0.0, dot(slice_dir, -gbuffer.normal));
            }

            //if (dir_i == 0)
            {
                scatter += total_radiance * weights[dir_i];
            }

            integr_tex[preintegr_px] = lerp(integr_tex[preintegr_px], float4(total_radiance, 0), blend_factor);
        } else {
            int3 src_px = int3(px) + dirs[dir_i];
            if (src_px.x >= 0 && src_px.x < GI_VOLUME_DIMS) {
                scatter += cascade0_tex[src_px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx].rgb;
            }

            integr_tex[preintegr_px] = lerp(integr_tex[preintegr_px], float4(0, 0, 0, 1), blend_factor);
        }
        
        /*if (!rt_is_shadowed(
            acceleration_structure,
            new_ray(
                trace_origin,
                neighbor_pos - trace_origin,
                0,
                1.0
        ))) {
            scatter += cascade0_tex[int3(px) + dirs[dir_i]].xyz * weights[dir_i];
        }*/
    }

    scatter = 0.0;
    total_wt = DIR_COUNT;

    for (uint i = 0; i < DIR_COUNT; ++i) {
        const int3 preintegr_px = px * int3(1, DIR_COUNT, 1) + int3(GI_VOLUME_DIMS * grid_idx, i, 0);
        const float4 color_transp = integr_tex[preintegr_px];

        scatter += color_transp.rgb;

        int3 src_px = int3(px) + dirs[i];
        if (src_px.x >= 0 && src_px.x < GI_VOLUME_DIMS) {
            scatter += cascade0_tex[src_px + int3(GI_VOLUME_DIMS * grid_idx, 0, 0)].rgb * color_transp.a;
        }
    }

    float3 radiance = scatter.xyz / total_wt;

    float4 prev = cascade0_tex[px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx];
    float4 cur = float4(radiance, 1);
    //float4 output = lerp(prev, cur, 0.2);
    float4 output = cur;
    cascade0_tex[px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx] = output;
}
