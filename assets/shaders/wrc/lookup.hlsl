#ifndef WRC_LOOKUP_HLSL
#define WRC_LOOKUP_HLSL

#include "wrc_settings.hlsl"
#include "../inc/pack_unpack.hlsl"

float4 lookup_wrc(int3 probe_coord, float3 dir) {
    const uint probe_idx = probe_coord_to_idx(probe_coord);
    const uint2 tile = wrc_probe_idx_to_atlas_tile(probe_idx);
    const float2 tile_uv = octa_encode(dir);
    const uint2 atlas_px = uint2((tile + tile_uv) * WRC_PROBE_DIMS);
    //return wrc_radiance_atlas_tex[atlas_px];
    const float2 atlas_uv = float2(tile + clamp(tile_uv, 0.5 / WRC_PROBE_DIMS, 1.0 - 0.5 / WRC_PROBE_DIMS)) / WRC_ATLAS_PROBE_COUNT;
    return wrc_radiance_atlas_tex.SampleLevel(sampler_llc, atlas_uv, 0);
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
    float inv_pdf;

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
    res.approx_surface_t = -1;
    res.inv_pdf = 1;
    res.radiance = 0;
    res.inv_pdf = 1;

    Aabb wrc_grid_box = Aabb::from_center_half_extent(wrc_grid_center(), WRC_GRID_WORLD_SIZE * 0.5);
    RayBoxIntersection wrc_grid_intersection = wrc_grid_box.intersect_ray(ray_origin, ray_direction);

    if (wrc_grid_intersection.is_hit()) {
        const float3 origin_in_box = ray_origin + ray_direction * max(0.0, wrc_grid_intersection.t_min);
        uint rng = hash4(uint4(ray_origin, frame_constants.frame_index));

        const float interpolation_extent = 1.0;
        float total_weight = 0;

        #define WRC_USE_STOCHASTIC_INTERPOLATION 0

    #if WRC_USE_STOCHASTIC_INTERPOLATION
        // Stochastic interpolation
        //const float3 probe_offset = (interpolation_urand - 0.5) * interpolation_extent;
        const float3 probe_offset = 0;
        {
            const float interp_weight = 1;
    #else
        for (int iz = 0; iz < 2; ++iz)
        for (int iy = 0; iy < 2; ++iy)
        for (int ix = 0; ix < 2; ++ix) {            
            const float3 probe_offset = float3(ix, iy, iz);
            const float3 interp_frac = lerp(1 - probe_offset, probe_offset, wrc_world_pos_to_interp_frac(origin_in_box));
            const float interp_weight = interp_frac.x * interp_frac.y * interp_frac.z;
    #endif

            const uint3 probe_coord = clamp(
                wrc_world_pos_to_coord(origin_in_box + probe_offset),
                int3(0, 0, 0),
                WRC_GRID_DIMS - 1
            );
            const float3 probe_center = wrc_probe_center(probe_coord);
            
            const Sphere probe_sphere = Sphere::from_center_radius(probe_center, WRC_MIN_TRACE_DIST);
            RaySphereIntersection parallax = probe_sphere.intersect_ray_inside(ray_origin, ray_direction);
            if (parallax.is_hit()) {
                total_weight += interp_weight;
            
                float4 out_value = lookup_wrc(probe_coord, parallax.normal);

                // With parallax correction, the query ray origin essentially moves
                // within a sphere centered around the probe. As the query center
                // does that, some texels of the probe will shrink, and others expand.
                // Here we adjust for this expansion, making sure energy does not get
                // biased towards the texels we're moving towards.
                const float distance_to_box = length(origin_in_box - ray_origin);
                const float3 parallax_pos = ray_origin + ray_direction * parallax.t;
                const float3 offset_from_query_pt = parallax_pos - ray_origin;
                const float3 offset_from_probe_center = parallax_pos - probe_center;
                const float parallax_dist2 = dot(offset_from_query_pt, offset_from_query_pt);

                // TODO: check all this
                float jacobian = 1;
                if (all(query_normal) != 0) {
                    jacobian *=
                        parallax_dist2 / pow(distance_to_box + WRC_MIN_TRACE_DIST, 2)
                        / dot(ray_direction, normalize(offset_from_probe_center));

                    // Also account for the change in the PDF being used in lighting.
                    // TODO: might want to move out to a place which knows the BRDF.
                    jacobian *=
                        dot(query_normal, normalize(offset_from_probe_center))
                        / dot(query_normal, ray_direction);
                }

                #if WRC_USE_STOCHASTIC_INTERPOLATION
                    res.radiance = out_value.rgb;
                    res.inv_pdf = max(0.0, jacobian);
                #else
                    // Olde. Not quite right by pushing the jacobian into radiance,
                    // but seems to work. Thought reservoir calculations would be affected...
                    res.radiance += out_value.rgb * max(0.0, jacobian) * interp_weight;
                    res.inv_pdf = 1;
                #endif

                #if WRC_USE_STOCHASTIC_INTERPOLATION
                    const float texel_footprint_fudge = 0.5;
                #else
                    const float texel_footprint_fudge = 0.25;
                #endif

                res.probe_t = max(res.probe_t, parallax.t + texel_footprint_fudge);
                res.approx_surface_t += parallax.t + out_value.a - WRC_MIN_TRACE_DIST * interp_weight;
            }
        }

        res.radiance /= total_weight;
        res.approx_surface_t /= total_weight;
    }

    return res;
}

#endif // WRC_LOOKUP_HLSL
