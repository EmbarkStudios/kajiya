#ifndef WRC_INTERSECT_PROBE_GRID_HLSL
#define WRC_INTERSECT_PROBE_GRID_HLSL

#include "lookup.hlsl"

float4 wrc_intersect_probe_grid(float3 ray_origin, float3 ray_dir, float max_t) {
    float4 hit_color = 0;

#if 1
    // DDA based on https://www.shadertoy.com/view/4dX3zl

    Aabb wrc_grid_box = Aabb::from_center_half_extent(wrc_grid_center(), WRC_GRID_WORLD_SIZE * 0.5);
    RayBoxIntersection wrc_grid_intersection = wrc_grid_box.intersect_ray(ray_origin, ray_dir);

    if (wrc_grid_intersection.is_hit()) {
        const float3 rayPos = ray_origin + ray_dir * max(0.0, wrc_grid_intersection.t_min);

        int3 coord = clamp(
            wrc_world_pos_to_coord(rayPos),
            int3(0, 0, 0),
            WRC_GRID_DIMS - 1
        );
        float3 delta_dist = abs(length(ray_dir) / ray_dir);
        int3 ray_step = int3(sign(ray_dir));
        float3 side_dist = (sign(ray_dir) * (wrc_probe_center(coord) - 0.5 - rayPos) + (sign(ray_dir) * 0.5) + 0.5) * delta_dist;
        bool3 mask;

        const int max_steps = WRC_GRID_DIMS.x + WRC_GRID_DIMS.y + WRC_GRID_DIMS.z;
        for (int step = 0; step < max_steps && all(coord == clamp(coord, int3(0, 0, 0), WRC_GRID_DIMS - 1)); ++step) {
            Sphere s = Sphere::from_center_radius(wrc_probe_center(coord), 0.1);
            RaySphereIntersection s_hit = s.intersect_ray(ray_origin, ray_dir);

            if (s_hit.is_hit()) {
                if (s_hit.t < max_t) {
                    const float3 refl = reflect(ray_dir, s_hit.normal);
                    hit_color = lookup_wrc(coord, refl);
                }

                break;
            }

            mask = side_dist.xyz <= min(side_dist.yzx, side_dist.zxy);
            side_dist += float3(mask) * delta_dist;
		    coord += int3(float3(mask)) * ray_step;
        }
    }
#else
    // Brute force

    float closest_hit = max_t;
    [loop] for (uint z = 0; z < WRC_GRID_DIMS.z; ++z) {
        [loop] for (uint y = 0; y < WRC_GRID_DIMS.y; ++y) {
            [loop] for (uint x = 0; x < WRC_GRID_DIMS.x; ++x) {
                const uint3 probe_coord = uint3(x, y, z);
                Sphere s = Sphere::from_center_radius(wrc_probe_center(probe_coord), 0.1);
                RaySphereIntersection s_hit = s.intersect_ray(ray_origin, ray_dir);

                if (s_hit.is_hit() && s_hit.t < closest_hit) {
                    closest_hit = s_hit.t;
                    //hit_color = float4(s_hit.normal * 0.5 + 0.5, 1);

                    const float3 refl = reflect(ray_dir, s_hit.normal);
                    hit_color = lookup_wrc(probe_coord, refl);
                }
            }
        }
    }
#endif

    return hit_color;
}

#endif