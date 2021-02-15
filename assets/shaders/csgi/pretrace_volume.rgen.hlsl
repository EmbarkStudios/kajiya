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

static const bool USE_MULTIBOUNCE = !true;

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0)]] RWTexture3D<float> out_hit_tex;
[[vk::binding(1)]] RWTexture3D<float4> out_col_tex;
[[vk::binding(2)]] RWTexture3D<float4> out_normal_tex;
[[vk::binding(3)]] Texture3D<float4> alt_cascade0_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 SLICE_DIRS[GI_SLICE_COUNT];
    float4 PRETRACE_DIRS[GI_PRETRACE_COUNT];
}

// HACK; broken
#define cascade0_tex alt_cascade0_tex
#include "lookup.hlsl"

static const float SKY_DIST = 1e5;

float3 vx_to_pos(float3 vx, float3x3 slice_rot) {
    return mul(slice_rot, vx - (GI_PRETRACE_DIMS - 1.0) / 2.0) * GI_VOXEL_SIZE + gi_volume_center(slice_rot);
}

[shader("raygeneration")]
void main() {
    uint2 px = DispatchRaysIndex().xy;
    const uint grid_idx = px.x / GI_PRETRACE_DIMS;
    px.x %= GI_PRETRACE_DIMS;

    const float3x3 slice_rot = build_orthonormal_basis(PRETRACE_DIRS[grid_idx].xyz);
    //const float3 slice_dir = mul(slice_rot, float3(0, 0, -1));

    int slice_z = GI_PRETRACE_DIMS - 1;

    while (true) {
        const int3 slice_px = int3(px, slice_z);
        const float3 trace_origin = vx_to_pos(float3(slice_px) + float3(0, 0, 0.5), slice_rot);

        RayDesc outgoing_ray;
        outgoing_ray.Direction = vx_to_pos(float3(slice_px) + float3(0, 0, -0.5), slice_rot) - trace_origin;
        outgoing_ray.Origin = trace_origin;
        outgoing_ray.TMin = 0;
        outgoing_ray.TMax = slice_z + 1;

        float3 hit_normal = 0.0.xxx;

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
                hit_normal = gbuffer.normal;

                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;

#if 0
                const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
                const float3 wi = mul(to_light_norm, shading_basis);
                float3 wo = normalize(mul(-outgoing_ray.Direction, shading_basis));

                LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);
                const float3 brdf_value = brdf.evaluate(wo, wi);
                total_radiance += brdf_value * light_radiance;
#else
                total_radiance +=
                    light_radiance * lerp(gbuffer.albedo, 1.0.xxx, 0.04) * max(0.0, dot(gbuffer.normal, to_light_norm)) / M_PI;
#endif

                if (USE_MULTIBOUNCE) {
                    CsgiLookupParams gi_lookup_params;
                    gi_lookup_params.use_grid_linear_fetch = false;
                    gi_lookup_params.use_pretrace = true;
                    gi_lookup_params.debug_slice_idx = -1;
                    //gi_lookup_params.slice_dirs = SLICE_DIRS;
                    //gi_lookup_params.cascade0_tex = cascade0_tex;
                    //gi_lookup_params.alt_cascade0_tex = alt_cascade0_tex;

                    total_radiance += lookup_csgi(primary_hit.position, gbuffer.normal, gi_lookup_params) * gbuffer.albedo;
                }
            }

            const int cells_skipped_by_ray = int(floor(primary_hit.ray_t));
            slice_z -= cells_skipped_by_ray + 1;

            if (slice_z >= -1) {
                // Write hit info in cells before the intersection
                out_col_tex[int3(px, slice_z + 1) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = float4(total_radiance, max(1e-5, primary_hit.ray_t - cells_skipped_by_ray));
                out_hit_tex[int3(px, slice_z + 1) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = 1;
                out_normal_tex[int3(px, slice_z + 1) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = float4(hit_normal * 0.5 + 0.5, 0.0);

                if (cells_skipped_by_ray > 0) {
                    out_col_tex[int3(px, slice_z + 2) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = float4(total_radiance, max(1e-5, primary_hit.ray_t - cells_skipped_by_ray + 1));
                    out_hit_tex[int3(px, slice_z + 2) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = 1;
                    out_normal_tex[int3(px, slice_z + 2) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = float4(hit_normal * 0.5 + 0.5, 0.0);
                }

                /*if (cells_skipped_by_ray > 1) {
                    out_col_tex[int3(px, slice_z + 3) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = float4(total_radiance, max(1e-5, primary_hit.ray_t - cells_skipped_by_ray + 2));
                    out_hit_tex[int3(px, slice_z + 3) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = 1;
                    out_normal_tex[int3(px, slice_z + 3) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = float4(hit_normal * 0.5 + 0.5, 0.0);
                }*/

                /*if (cells_skipped_by_ray > 2) {
                    out_col_tex[int3(px, slice_z + 4) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = float4(total_radiance, max(1e-5, primary_hit.ray_t - cells_skipped_by_ray + 3));
                    out_hit_tex[int3(px, slice_z + 4) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = 1;
                }*/
            } else {
                //out_col_tex[int3(px, 0) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = float4(total_radiance, 0);
                //out_hit_tex[int3(px, 0) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = 1;
                //out_normal_tex[int3(px, 0) + int3(GI_PRETRACE_DIMS, 0, 0) * grid_idx] = float4(hit_normal * 0.5 + 0.5, 0.0);
                return;
            }
        } else {
            return;
        }
    }
}
