#include "sdf/sdf_common.hlsl"

[[vk::binding(1)]] Texture3D<float> sdf_tex;
[[vk::binding(2)]] SamplerState sampler_nnc;
[[vk::binding(3)]] SamplerState sampler_lnc;

struct PsIn {
    [[vk::location(0)]] float4 color: COLOR0;
    [[vk::location(1)]] float3 vs_pos: COLOR1;

    // TODO: nointerp
    [[vk::location(2)]] nointerpolation float4 cell_pos_extent: COLOR2;
};

bool is_inside_volume(float3 p, float4 cell_pos_extent) {
    p -= cell_pos_extent.xyz;
    float eps = 1e-3;
    float radius = cell_pos_extent.w * 0.5 * (1 + eps);
    return all(abs(p) < radius);
}

static float3 g_mouse_pos;
float sample_volume(float3 p, SamplerState smp) {
    float3 uv = (p / HSIZE / 2.0) + 0.5.xxx;
    return sdf_tex.SampleLevel(smp, uv, 0);
}

float4 main(PsIn ps/*, float4 cs_pos: SV_Position*/): SV_TARGET {
#if 1
    ViewConstants view_constants = frame_constants.view_constants;

    float4 cs_pos = mul(view_constants.view_to_sample, float4(ps.vs_pos, 1));
    cs_pos /= cs_pos.w;
    float4 ray_origin_cs = cs_pos;
    float4 ray_origin_ws = mul(view_constants.view_to_world, float4(ps.vs_pos, 1));
    ray_origin_ws /= ray_origin_ws.w;

    float4 ray_dir_cs = float4(ray_origin_cs.xy, 0.0, 1.0);
    float4 ray_dir_ws = mul(view_constants.view_to_world, mul(view_constants.sample_to_view, ray_dir_cs));
    float3 v = -normalize(ray_dir_ws.xyz);

    const uint ITERS = 32;

    float3 p = ray_origin_ws.xyz;
    float dist = sample_volume(p, sampler_lnc);
    p += -v * max(0.0, dist);

    for (uint iter = 0; iter < ITERS; ++iter) {
        if (dist < 0.0 || !is_inside_volume(p, ps.cell_pos_extent)) {
            break;
        } else {
            dist = sample_volume(p, sampler_lnc);
            p += -v * max(0.0, dist);
        }
    }

    float4 res = float4(1.0, 0.0, 1.0, 1);
    
    if (is_inside_volume(p, ps.cell_pos_extent)) {
        float3 uv = (p / HSIZE / 2.0) + 0.5.xxx;
        float dstep = 2.0 * HSIZE / SDFRES;
        float dx = sample_volume(p + float3(dstep, 0, 0), sampler_lnc);
        float dy = sample_volume(p + float3(0, dstep, 0), sampler_lnc);
        float dz = sample_volume(p + float3(0, 0, dstep), sampler_lnc);

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
    } else {
        discard;
    }

    return res;
#else
    return ps.color;
#endif
}
