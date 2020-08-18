#include "sdf_consts.inc"

vec3 intersect_ray_plane(vec3 normal, vec3 plane_pt, vec3 o, vec3 dir) {
    return o - dir * (dot(o - plane_pt, normal) / dot(dir, normal));
}

vec3 get_sdf_brush_pos() {
    vec3 eye_pos_ws = (view_constants.view_to_world * vec4(0, 0, 0, 1)).xyz;
    vec3 eye_dir_ws = normalize((view_constants.view_to_world * (view_constants.sample_to_view * vec4(0.0, 0.0, 0.0, 1.0))).xyz);
    vec4 mouse_dir_cs = vec4(uv_to_cs(mouse.xy), 0.0, 1.0);
    vec4 mouse_dir_ws = view_constants.view_to_world * (view_constants.sample_to_view * mouse_dir_cs);
    vec3 mouse_pos = intersect_ray_plane(eye_dir_ws, eye_pos_ws + eye_dir_ws * 8.0, eye_pos_ws, mouse_dir_ws.xyz);
    return mouse_pos;
}

float get_sdf_brush_radius() {
    return 0.4;
}
