#include "rendertoy::shaders/view_constants.inc"
#include "rtoy-samples::shaders/inc/uv.inc"

layout(r16f) uniform restrict image3D outputTex;

layout(std430) buffer constants {
    ViewConstants view_constants;
    vec4 mouse;
};

layout(std430) buffer dispatch_params {
    uint groupsX;
    uint groupsY;
    uint groupsZ;
    uint pad0;
    int offsetX;
    int offsetY;
    int offsetZ;
    uint pad1;
};

#include "sdf_common.inc"

vec3 ws_pos;

float sd_sphere(vec3 p, float s) {
  return length(p) - s;
}

float op_sub(float d1, float d2) {
    return max(-d1, d2);
}

float op_union(float d1, float d2) {
    return min(d1, d2);
}

layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main() {
    ivec3 pix = ivec3(gl_GlobalInvocationID.xyz) + ivec3(offsetX, offsetY, offsetZ);
    ws_pos = ((vec3(pix) / vec3(SDFRES, SDFRES, SDFRES)) - 0.5) * 2.0 * HSIZE;

    float result = 0.5;

    if (mouse.z > 0.0) {
        vec3 mouse_pos = get_sdf_brush_pos();
        result = op_union(result, sd_sphere(ws_pos - mouse_pos, get_sdf_brush_radius()));
    }
    
    //result = op_union(s1, s0);

    /*if (frame_idx % 2 == 0) {
        result = op_union(s1, s0);
    }*/

    float prev = imageLoad(outputTex, pix).x;

    if (mouse.w > 0.0) {
        result = op_union(result, prev);
    } else {
        result = op_sub(result, prev);
    }

    imageStore(outputTex, pix, result.xxxx);
}
