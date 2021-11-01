#ifndef WRC_LOOKUP_HLSL
#define WRC_LOOKUP_HLSL

#include "wrc_settings.hlsl"
#include "../inc/pack_unpack.hlsl"

float4 lookup_wrc(int3 probe_coord, float3 dir) {
    const uint probe_idx = probe_coord_to_idx(probe_coord);
    const uint2 tile = wrc_probe_idx_to_atlas_tile(probe_idx);
    const float2 tile_uv = octa_encode(dir);
    const uint2 atlas_px = uint2((tile + tile_uv) * WRC_PROBE_DIMS);
    return wrc_radiance_atlas_tex[atlas_px];
}

struct WrcFarField;
struct WrcFarFieldQuery {
    float3 ray_origin;
    float3 ray_direction;
    float3 interpolation_urand;
    float3 query_normal;

    static WrcFarFieldQuery from_ray(float3 ray_origin, float3 ray_direction) {
        WrcFarFieldQuery res;
        res.ray_origin = ray_origin;
        res.ray_direction = ray_direction;
        res.interpolation_urand = 0.5.xxx;
        res.query_normal = 0.0.xxx;
        return res;
    }

    WrcFarFieldQuery with_interpolation_urand(float3 interpolation_urand) {
        WrcFarFieldQuery res = this;
        res.interpolation_urand = interpolation_urand;
        return res;
    }

    WrcFarFieldQuery with_query_normal(float3 query_normal) {
        WrcFarFieldQuery res = this;
        res.query_normal = query_normal;
        return res;
    }

    WrcFarField query();
};

struct WrcFarField {
    float3 radiance;
    float probe_t;
    float approx_surface_t;

    static WrcFarField create_miss() {
        WrcFarField res;
        res.probe_t = -1;
        return res;
    }

    bool is_hit() {
        return probe_t >= 0;
    }
};

WrcFarField WrcFarFieldQuery::query() {
    WrcFarField res;
    res.probe_t = -1;

    Aabb wrc_grid_box = Aabb::from_center_half_extent(0.0.xxx, WRC_GRID_WORLD_SIZE * 0.5);
    RayBoxIntersection wrc_grid_intersection = wrc_grid_box.intersect_ray(ray_origin, ray_direction);

    if (wrc_grid_intersection.is_hit()) {
        const float3 origin_in_box = ray_origin + ray_direction * max(0.0, wrc_grid_intersection.t_min);
        uint rng = hash4(uint4(ray_origin, frame_constants.frame_index));

        // Stochastic interpolation
        const float interpolation_extent = 1.0;
        const float3 probe_offset = (interpolation_urand - 0.5) * interpolation_extent;

        const uint3 probe_coord = clamp(
            wrc_world_pos_to_coord(origin_in_box + probe_offset),
            int3(0, 0, 0),
            WRC_GRID_DIMS - 1
        );
        const float3 probe_center = wrc_probe_center(probe_coord);
        
        const Sphere probe_sphere = Sphere::from_center_radius(probe_center, WRC_MIN_TRACE_DIST);
        RaySphereIntersection parallax = probe_sphere.intersect_ray_inside(ray_origin, ray_direction);
        if (parallax.is_hit()) {
            float4 out_value = lookup_wrc(probe_coord, parallax.normal);

            // With parallax correction, the query ray origin essentially moves
            // within a sphere centered around the probe. As the query center
            // does that, some texels of the probe will shrink, and others expand.
            // Here we adjust for this expansion, making sure energy does not get
            // biased towards the texels we're moving towards.
            const float3 parallax_pos = ray_origin + ray_direction * parallax.t;
            const float3 offset_from_query_pt = parallax_pos - origin_in_box;
            const float3 offset_from_probe_center = parallax_pos - probe_center;
            const float parallax_dist2 = dot(offset_from_query_pt, offset_from_query_pt);

            float jacobian = 1;
            if (all(query_normal) != 0) {
                jacobian =
                    parallax_dist2 / (WRC_MIN_TRACE_DIST * WRC_MIN_TRACE_DIST)
                    / dot(ray_direction, normalize(offset_from_probe_center));

                // Also account for the change in the PDF being used in lighting.
                // TODO: do all of those terms make sense?
                jacobian *=
                    dot(query_normal, normalize(offset_from_probe_center))
                    / dot(query_normal, ray_direction);
            }

            res.radiance = out_value.rgb * jacobian;

            const float texel_footprint_fudge = 0.5;
            res.probe_t = parallax.t + texel_footprint_fudge;

            res.approx_surface_t = parallax.t + out_value.a - WRC_MIN_TRACE_DIST;
        }
    }

    return res;
}

#endif // WRC_LOOKUP_HLSL
