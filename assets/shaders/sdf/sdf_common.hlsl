#include "../inc/frame_constants.hlsl"
#include "../inc/uv.hlsl"
#include "sdf_consts.hlsl"

float3 intersect_ray_plane(float3 normal, float3 plane_pt, float3 o, float3 dir) {
    return o - dir * (dot(o - plane_pt, normal) / dot(dir, normal));
}

float3 get_sdf_brush_pos() {
    ViewConstants view_constants = frame_constants.view_constants;
    float3 eye_pos_ws = mul(view_constants.view_to_world, float4(0, 0, 0, 1)).xyz;
    float3 eye_dir_ws = normalize(mul(view_constants.view_to_world, mul(view_constants.sample_to_view, float4(0.0, 0.0, 0.0, 1.0))).xyz);
    float4 mouse_dir_cs = float4(uv_to_cs(frame_constants.mouse.xy), 0.0, 1.0);
    float4 mouse_dir_ws = mul(view_constants.view_to_world, mul(view_constants.sample_to_view, mouse_dir_cs));
    float3 mouse_pos = intersect_ray_plane(eye_dir_ws, eye_pos_ws + eye_dir_ws * 8.0, eye_pos_ws, mouse_dir_ws.xyz);
    return mouse_pos;
}

float get_sdf_brush_radius() {
    return 0.4;
}

float sd_sphere(float3 p, float s) {
  return length(p) - s;
}

float op_sub(float d1, float d2) {
    return max(-d1, d2);
}

float op_union(float d1, float d2) {
    return min(d1, d2);
}
