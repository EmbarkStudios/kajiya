#include "../inc/math.hlsl"

void get_surfel_grid_box_min_max(Vertex surfel, out int3 box_min, out int3 box_max) {
    #if 0
        float3x3 xform = abs(build_orthonormal_basis(surfel.normal));

        zmax = (r23 + sqrt(pow(r23,2) - (r33*r22)) ) / r33; 
      	zmin = (r23 - sqrt(pow(r23,2) - (r33*r22)) ) / r33; 

      	ymax = (r13 + sqrt(pow(r13,2) - (r33*r11)) ) / r33; 
      	ymin = (r13 - sqrt(pow(r13,2) - (r33*r11)) ) / r33; 
      	xmax = (r03 + sqrt(pow(r03,2) - (r33*r00)) ) / r33; 
      	xmin = (r03 - sqrt(pow(r03,2) - (r33*r00)) ) / r33; 

        const float3 box_min_pos = surfel.position - obb_extent;
        const float3 box_max_pos = surfel.position + obb_extent;
    #else
        const float3 box_min_pos = surfel.position - SURFEL_RADIUS;
        const float3 box_max_pos = surfel.position + SURFEL_RADIUS;
    #endif

    box_min = surfel_pos_to_grid_coord(box_min_pos);
    box_max = surfel_pos_to_grid_coord(box_max_pos);
}

bool surfel_intersects_grid_coord(Vertex surfel, int3 grid_coord) {
    const float3 cell_center = surfel_grid_coord_center(grid_coord);
    const float grid_cell_radius = SURFEL_GRID_CELL_DIAMETER * 0.5;

    const float3 cell_local_surfel_pos = surfel.position - cell_center;
    const float3 cell_local_closest_point_on_grid_cell =
        clamp(cell_local_surfel_pos, -grid_cell_radius, grid_cell_radius);

    const float3 pos_offset = cell_local_surfel_pos - cell_local_closest_point_on_grid_cell;

    #if 1
        // Approximate box-ellipsoid culling. Sometimes misses corners,
        // but greatly improves culling efficiency.
        // TODO: figure out a precise solution. Warp space before finding the closest point?
        const float mahalanobis_dist = length(pos_offset) * (1 + abs(dot(pos_offset, surfel.normal)) * SURFEL_NORMAL_DIRECTION_SQUISH);
        return mahalanobis_dist < SURFEL_RADIUS;
    #else
        return dot(pos_offset, pos_offset) < SURFEL_RADIUS * SURFEL_RADIUS;
    #endif
}
