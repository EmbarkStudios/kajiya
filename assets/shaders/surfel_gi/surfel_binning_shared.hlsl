#include "../inc/math.hlsl"

struct SurfelGridMinMax {
    uint4 c4_min[2];
    uint4 c4_max[2];
    uint cascade_count;
};

SurfelGridMinMax get_surfel_grid_box_min_max(Vertex surfel) {
    const float surfel_radius = surfel_radius_for_pos(surfel.position);

    // TODO: OBB around normal?
    const float3 box_min_pos = surfel.position - surfel_radius;
    const float3 box_max_pos = surfel.position + surfel_radius;

    const float3 eye_pos = get_eye_position();
    const float fc = surfel_grid_coord_to_cascade_float(surfel_pos_to_grid_coord(surfel.position, eye_pos));

    // TODO: bounds
    const uint c0 = surfel_cascade_float_to_cascade(fc - 0.2);
    const uint c1 = surfel_cascade_float_to_cascade(fc + 0.2);

    const int3 min_coord = surfel_pos_to_grid_coord(box_min_pos, eye_pos);
    const int3 max_coord = surfel_pos_to_grid_coord(box_max_pos, eye_pos);

    SurfelGridMinMax result;

    result.cascade_count = 1;
    result.c4_min[0] = uint4(
        clamp(surfel_grid_coord_within_cascade(min_coord, c0), (int3)(0), (int3)(SURFEL_CS - 1)),
        c0);
    result.c4_max[0] = uint4(
        clamp(surfel_grid_coord_within_cascade(max_coord, c0), (int3)(0), (int3)(SURFEL_CS - 1)),
        c0);

    if (c1 != c0) {
        result.cascade_count = 2;
        result.c4_min[1] = uint4(
            clamp(surfel_grid_coord_within_cascade(min_coord, c1), (int3)(0), (int3)(SURFEL_CS - 1)),
            c1);
        result.c4_max[1] = uint4(
            clamp(surfel_grid_coord_within_cascade(max_coord, c1), (int3)(0), (int3)(SURFEL_CS - 1)),
            c1);
    }

    return result;
}

bool surfel_intersects_grid_coord(Vertex surfel, uint4 grid_coord) {
    const float surfel_radius = surfel_radius_for_pos(surfel.position);

    const float3 cell_center = surfel_grid_coord_center(grid_coord, get_eye_position());
    const float grid_cell_radius = (SURFEL_GRID_CELL_DIAMETER * 0.5) * (1u << grid_coord.w);

    const float3 cell_local_surfel_pos = surfel.position - cell_center;
    const float3 cell_local_closest_point_on_grid_cell =
        clamp(cell_local_surfel_pos, -grid_cell_radius, grid_cell_radius);

    const float3 pos_offset = cell_local_surfel_pos - cell_local_closest_point_on_grid_cell;

    #if 1
        // Approximate box-ellipsoid culling. Sometimes misses corners,
        // but greatly improves culling efficiency.
        // TODO: figure out a precise solution. Warp space before finding the closest point?
        const float mahalanobis_dist = length(pos_offset) * (1 + abs(dot(pos_offset, surfel.normal)) * SURFEL_NORMAL_DIRECTION_SQUISH);
        return mahalanobis_dist < surfel_radius;
    #else
        return dot(pos_offset, pos_offset) < surfel_radius * surfel_radius;
    #endif
}
