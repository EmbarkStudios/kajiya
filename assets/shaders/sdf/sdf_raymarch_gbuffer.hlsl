//#include "rendertoy::shaders/view_constants.inc"
//#include "rtoy-samples::shaders/inc/uv.inc"
//#include "rtoy-samples::shaders/inc/pack_unpack.inc"
#include "sdf_consts.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/uv.hlsl"

[[vk::binding(0)]] RWTexture2D<float4> output_tex;
[[vk::binding(1)]] Texture3D<float> sdf_tex;
[[vk::binding(2)]] SamplerState sampler_lnc;

bool is_inside_volume(float3 p) {
    return abs(p.x) < HSIZE && abs(p.y) < HSIZE && abs(p.z) < HSIZE;
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

static float3 mouse_pos;
float sample_volume(float3 p) {
    float3 uv = (p / HSIZE / 2.0) + 0.5.xxx;
    float d0 = sdf_tex.SampleLevel(sampler_lnc, uv, 0);
    float d1 = sd_sphere(p - mouse_pos, 0.4);
    if (frame_constants.mouse.w > 0.0) {
        return op_union(d0, d1);
    } else {
        return op_sub(d1, d0);
    }
}

float3 intersect_ray_plane(float3 normal, float3 plane_pt, float3 o, float3 dir) {
    return o - dir * (dot(o - plane_pt, normal) / dot(dir, normal));
}

[numthreads(8, 8, 1)]
void main(in uint2 pix : SV_DispatchThreadID) {
    #if 1
    float4 output_tex_size = float4(1280, 720, 1.0 / 1280, 1.0 / 720);
    ViewConstants view_constants = frame_constants.view_constants;

    float2 uv = get_uv(pix, output_tex_size);

    float4 ray_origin_cs = float4(uv_to_cs(uv), 1.0, 1.0);
    float4 ray_origin_ws = mul(view_constants.view_to_world, mul(view_constants.sample_to_view, ray_origin_cs));
    ray_origin_ws /= ray_origin_ws.w;

    float4 ray_dir_cs = float4(uv_to_cs(uv), 0.0, 1.0);
    float4 ray_dir_ws = mul(view_constants.view_to_world, mul(view_constants.sample_to_view, ray_dir_cs));
    float3 v = -normalize(ray_dir_ws.xyz);

    float3 eye_pos_ws = mul(view_constants.view_to_world, float4(0, 0, 0, 1)).xyz;
    float3 eye_dir_ws = normalize(mul(view_constants.view_to_world, mul(view_constants.sample_to_view, float4(0.0, 0.0, 0.0, 1.0))).xyz);
    float4 mouse = frame_constants.mouse;
    float4 mouse_dir_cs = float4(uv_to_cs(mouse.xy), 0.0, 1.0);
    float4 mouse_dir_ws = mul(view_constants.view_to_world, mul(view_constants.sample_to_view, mouse_dir_cs));
    mouse_pos = intersect_ray_plane(eye_dir_ws, eye_pos_ws + eye_dir_ws * 8.0, eye_pos_ws, mouse_dir_ws.xyz);

    const uint ITERS = 128;
    float dist = 1.0;

    float3 p = ray_origin_ws.xyz;

    if (!is_inside_volume(p)) {
        float3 d = ((HSIZE - 0.01).xxx * sign(v) - p) / -v;
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

    //float4 res = 0.0.xxxx;
    float4 res = float4(0.1, 0.2, 0.5, 1);
    
    if (is_inside_volume(p)) {
        float3 uv = (p / HSIZE / 2.0) + 0.5.xxx;
        float dstep = 2.0 * HSIZE / SDFRES;
        float dx = sample_volume(p + float3(dstep, 0, 0));
        float dy = sample_volume(p + float3(0, dstep, 0));
        float dz = sample_volume(p + float3(0, 0, dstep));

        float3 normal = normalize(float3(dx, dy, dz));
        /*float roughness = 0.1;
        float3 albedo = 0.5.xxx;
        float4 p_cs = view_constants.view_to_sample * (view_constants.world_to_view * float4(p, 1));
        float z_over_w = p_cs.z / p_cs.w;

        res.x = pack_normal_11_10_11(normal);
        res.y = roughness * roughness;      // UE4 remap
        res.z = uintBitsToFloat(pack_color_888(albedo));
        res.w = z_over_w;*/
        res = float4(pow(normal * 0.5 + 0.5, 2), 1);
    }

    output_tex[pix] = res;
    #else
        output_tex[pix] = sample_volume(float3(0, 0, 0)).xxxx;
    #endif
}
