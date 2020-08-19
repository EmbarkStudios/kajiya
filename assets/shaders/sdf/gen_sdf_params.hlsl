#include "sdf_common.hlsl"

layout(std430) buffer outputBuf {
    uint groupsX;
    uint groupsY;
    uint groupsZ;
    uint pad0;
    int offsetX;
    int offsetY;
    int offsetZ;
    uint pad1;
};

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    if (mouse.z > 0.0) {
        // TODO: clamp dispatch size to volume extents

        float radius = get_sdf_brush_radius() + SDF_EMPTY_DIST;
        int brush_radius_tiles = int(radius / HSIZE * 0.5 * float(SDFRES));

        uint threads = uint(brush_radius_tiles * 2);
        uint groups = (threads + 3) / 4;
        groupsX = groups;
        groupsY = groups;
        groupsZ = groups;

        vec3 mouse_pos = get_sdf_brush_pos();
        mouse_pos /= vec3(HSIZE);
        mouse_pos *= 0.5;
        mouse_pos += 0.5;
        mouse_pos *= vec3(SDFRES);

        offsetX = int(mouse_pos.x) - brush_radius_tiles;
        offsetY = int(mouse_pos.y) - brush_radius_tiles;
        offsetZ = int(mouse_pos.z) - brush_radius_tiles;
    } else {
        groupsX = 0;
        groupsY = 0;
        groupsZ = 0;
        offsetX = 0;
        offsetY = 0;
        offsetZ = 0;
    }
}
