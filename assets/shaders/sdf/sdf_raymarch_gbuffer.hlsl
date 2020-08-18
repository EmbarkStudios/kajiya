//#include "rendertoy::shaders/view_constants.inc"
//#include "rtoy-samples::shaders/inc/uv.inc"
//#include "rtoy-samples::shaders/inc/pack_unpack.inc"
#include "sdf_consts.hlsl"

Texture3D<float> sdf_tex;
RWTexture2D<float4> output_tex;

[numthreads(8, 8, 1)]
void main(in uint2 pix : SV_DispatchThreadID) {
    output_tex[pix] = float4(1, 0, 0, 1);
}


/*vec3 sample_lambert_convolved_environment_light(vec3 dir) {
    dir = normalize(dir);
    return texelFetch(skyLambertTex, ivec2(skyLambertTex_size.xy * octa_encode(dir)), 0).rgb;
}

bool is_inside_volume(vec3 p) {
    return abs(p.x) < HSIZE && abs(p.y) < HSIZE && abs(p.z) < HSIZE;
}

float sd_sphere(vec3 p, float s) {
  return length(p) - s;
}

float op_sub(float d1, float d2) {
    return max(-d1, d2);
}

float op_union(float d1, float d2) {
    return min(d1, d2);
}

vec3 mouse_pos;
float sample_volume(vec3 p) {
    vec3 uv = (p / HSIZE / 2.0) + 0.5.xxx;
    float d0 = textureLod(sampler3D(sdf_tex, linear_clamp_sampler), uv, 0.0).r;
    float d1 = sd_sphere(p - mouse_pos, 0.4);
    if (mouse.w > 0.0) {
        return op_union(d0, d1);
    } else {
        return op_sub(d1, d0);
    }
}

vec3 intersect_ray_plane(vec3 normal, vec3 plane_pt, vec3 o, vec3 dir) {
    return o - dir * (dot(o - plane_pt, normal) / dot(dir, normal));
}

layout (local_size_x = 8, local_size_y = 8) in;
void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = get_uv(outputTex_size);
    vec3 result = textureLod(sampler2D(sky_tex, linear_clamp_sampler), uv, 0.0).rgb;

    vec4 ray_origin_cs = vec4(uv_to_cs(uv), 1.0, 1.0);
    vec4 ray_origin_ws = view_constants.view_to_world * (view_constants.sample_to_view * ray_origin_cs);
    ray_origin_ws /= ray_origin_ws.w;

    vec4 ray_dir_cs = vec4(uv_to_cs(uv), 0.0, 1.0);
    vec4 ray_dir_ws = view_constants.view_to_world * (view_constants.sample_to_view * ray_dir_cs);
    vec3 v = -normalize(ray_dir_ws.xyz);

    vec3 eye_pos_ws = (view_constants.view_to_world * vec4(0, 0, 0, 1)).xyz;
    vec3 eye_dir_ws = normalize((view_constants.view_to_world * (view_constants.sample_to_view * vec4(0.0, 0.0, 0.0, 1.0))).xyz);
    vec4 mouse_dir_cs = vec4(uv_to_cs(mouse.xy), 0.0, 1.0);
    vec4 mouse_dir_ws = view_constants.view_to_world * (view_constants.sample_to_view * mouse_dir_cs);
    mouse_pos = intersect_ray_plane(eye_dir_ws, eye_pos_ws + eye_dir_ws * 8.0, eye_pos_ws, mouse_dir_ws.xyz);

    const uint ITERS = 128;
    float dist = 1.0;

    vec3 p = ray_origin_ws.xyz;

    if (!is_inside_volume(p)) {
        vec3 d = ((HSIZE - 0.01).xxx * sign(v) - p) / -v;
        p += max(0.0, max(max(d.x, d.y), d.z)) * -v;
    }

    for (uint iter = 0; iter < ITERS; ++iter) {
        if (dist < 0.0 || !is_inside_volume(p)) {
            break;
        } else {
            dist = sample_volume(p);
            p += -v * max(0.0, dist);
        }
    }

    vec4 res = 0.0.xxxx;
    
    if (is_inside_volume(p)) {
        vec3 uv = (p / HSIZE / 2.0) + 0.5.xxx;
        float dstep = 2.0 * HSIZE / SDFRES;
        float dx = sample_volume(p + vec3(dstep, 0, 0));
        float dy = sample_volume(p + vec3(0, dstep, 0));
        float dz = sample_volume(p + vec3(0, 0, dstep));

        vec3 normal = normalize(vec3(dx, dy, dz));
        float roughness = 0.1;
        vec3 albedo = 0.5.xxx;
        vec4 p_cs = view_constants.view_to_sample * (view_constants.world_to_view * vec4(p, 1));
        float z_over_w = p_cs.z / p_cs.w;

        res.x = pack_normal_11_10_11(normal);
        res.y = roughness * roughness;      // UE4 remap
        res.z = uintBitsToFloat(pack_color_888(albedo));
        res.w = z_over_w;
    }

    imageStore(outputTex, pix, res);
}
*/