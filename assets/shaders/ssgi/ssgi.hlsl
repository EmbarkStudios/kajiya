#include "../inc/math_const.hlsl"
#include "../inc/math.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/gbuffer.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
//[[vk::binding(1)]] Texture2D<float4> reprojectedLightingTex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> view_normal_tex;
[[vk::binding(3)]] RWTexture2D<float4> output_tex;

[[vk::binding(4)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

static const float temporal_rotations[] = { 60.0, 300.0, 180.0, 240.0, 120.0, 0.0 };
static const float temporal_offsets[] = { 0.0, 0.5, 0.25, 0.75 };
static const uint ssgi_half_sample_count = 16;

float fast_sqrt(float x) {
    return asfloat(0x1fbd1df5 + (asuint(x) >> 1u));
}

// max absolute error 9.0x10^-3
// Eberly's polynomial degree 1 - respect bounds
// 4 VGPR, 12 FR (8 FR, 1 QR), 1 scalar
// input [-1, 1] and output [0, M_PI]
float fast_acos(float inX) 
{ 
    float x = abs(inX); 
    float res = -0.156583f * x + (M_FRAC_PI_2); 
    res *= fast_sqrt(1.0f - x); 
    return (inX >= 0) ? res : M_PI - res;
}

struct Ray {
	float3 o;
	float3 d;
};

float fetch_depth(float2 uv) {
    return depth_tex[int2(input_tex_size.xy * uv)].x;
}

float3 fetch_lighting(float2 uv) {
    return 0.1.xxx;
    //return texelFetch(reprojectedLightingTex, int2(input_tex_size.xy * uv), 0).xyz;
}

float3 fetch_normal_vs(float2 uv) {
    int2 px = int2(output_tex_size.xy * uv);
    float3 normal_vs = view_normal_tex[px].xyz;
    return normal_vs;
}

float integrate_half_arc(float h1, float n) {
    float a = -cos(2.0 * h1 - n) + cos(n) + 2.0 * h1 * sin(n);
    return 0.25 * a;
}

float integrate_arc(float h1, float h2, float n) {
    float a = -cos(2.0 * h1 - n) + cos(n) + 2.0 * h1 * sin(n);
    float b = -cos(2.0 * h2 - n) + cos(n) + 2.0 * h2 * sin(n);
    return 0.25 * (a + b);
}

float update_horizion_angle(float prev, float cur) {
    float t = min(1.0, 0.5 / float(ssgi_half_sample_count));
    return cur > prev ? max(cur, prev) : lerp(prev, cur, t);
    //return cur > prev ? max(cur, prev) : prev;
}

float intersect_dir_plane_onesided(float3 dir, float3 normal, float3 pt) {
    float d = -dot(pt, normal);
    float t = d / max(1e-5, -dot(dir, normal));
    return t;
}

float3 project_point_on_plane(float3 pt, float3 normal) {
    return pt - normal * dot(pt, normal);
}

float process_sample(uint i, float intsgn, float n_angle, inout float3 prev_sample_vs, float4 sample_cs, float3 center_vs, float3 normal_vs, float3 v_vs, float ao_radius, float theta_cos_max, inout float4 color_accum) {
    if (sample_cs.z > 0) {
        float4 sample_vs4 = mul(frame_constants.view_constants.sample_to_view, sample_cs);
        float3 sample_vs = sample_vs4.xyz / sample_vs4.w;
        float3 sample_vs_offset = sample_vs - center_vs;
        float sample_vs_offset_len = length(sample_vs_offset);

        float sample_theta_cos = dot(sample_vs_offset, v_vs) / sample_vs_offset_len;
        if (sample_vs_offset_len < ao_radius) {
            bool sample_visible = sample_theta_cos >= theta_cos_max;
            float theta_cos_prev = theta_cos_max;
            float theta_delta = theta_cos_max;
            theta_cos_max = update_horizion_angle(theta_cos_max, sample_theta_cos);
            theta_delta = theta_cos_max - theta_delta;

            if (sample_visible) {
                float3 lighting = fetch_lighting(cs_to_uv(sample_cs.xy));

                float3 sample_normal_vs = fetch_normal_vs(cs_to_uv(sample_cs.xy));
                float theta_cos_prev_trunc = theta_cos_prev;

#if 1
                {
                    float3 p1 = prev_sample_vs * min(
                        intersect_dir_plane_onesided(prev_sample_vs, sample_normal_vs, sample_vs),
                        intersect_dir_plane_onesided(prev_sample_vs, normal_vs, center_vs)
                    );

                    theta_cos_prev_trunc = clamp(dot(normalize(p1 - center_vs), v_vs), theta_cos_prev_trunc, theta_cos_max);
                }
#endif
#if 1
                n_angle *= -intsgn;

                float h1 = fast_acos(theta_cos_prev_trunc);
                float h2 = fast_acos(theta_cos_max);

                float h1p = n_angle + max(h1 - n_angle, -M_FRAC_PI_2);
                float h2p = n_angle + min(h2 - n_angle, M_FRAC_PI_2);

                float inv_ao =
                    integrate_half_arc(h1p, n_angle) -
                    integrate_half_arc(h2p, n_angle);
                    
                lighting *= inv_ao;
                lighting *= step(0.0, dot(-normalize(sample_vs_offset), sample_normal_vs));
#endif

                color_accum += float4(lighting, 1.0);
            }
        }

        prev_sample_vs = sample_vs;
    } else {
        // Sky; assume no occlusion
        theta_cos_max = update_horizion_angle(theta_cos_max, -1);
    }

    return theta_cos_max;
}

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);

    float4 gbuffer_packed = gbuffer_tex[px * 2];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        output_tex[px] = 1.0.xxxx;
        return;
    }

    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    const float3 normal_vs = normalize(mul(frame_constants.view_constants.world_to_view, float4(gbuffer.normal, 0)).xyz);

    float4 col = 0.0.xxxx;
    float ao_radius = 0.3;

    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, fetch_depth(uv));
    float3 v_vs = -normalize(view_ray_context.ray_dir_vs());

    float4 ray_hit_cs = view_ray_context.ray_hit_cs;
    float3 ray_hit_vs = view_ray_context.ray_hit_vs();
    
    float spatial_direction_noise = 1.0 / 16.0 * ((((px.x + px.y) & 3) << 2) + (px.x & 3));
    float temporal_direction_noise = temporal_rotations[frame_constants.frame_index % 6] / 360.0;
    float spatial_offset_noise = (1.0 / 4.0) * ((px.y - px.x) & 3);
    float temporal_offset_noise = temporal_offsets[frame_constants.frame_index / 6 % 4];

    uint seed0 = hash3(uint3(frame_constants.frame_index, px.x, px.y));
    spatial_direction_noise += uint_to_u01_float(seed0) * 0.1;

    float ss_angle = frac(spatial_direction_noise + temporal_direction_noise) * M_PI;
    float rand_offset = frac(spatial_offset_noise + temporal_offset_noise);

    float2 cs_slice_dir = float2(cos(ss_angle) * input_tex_size.y / input_tex_size.x, sin(ss_angle));

    float ao_radius_shrinkage;
    {
        // Convert AO radius into world scale
        float cs_ao_radius_rescale = ao_radius * frame_constants.view_constants.view_to_clip[1][1] / -ray_hit_vs.z;
        cs_slice_dir *= cs_ao_radius_rescale;

        // TODO: better units (pxels? degrees?)
        // Calculate AO radius shrinkage (if camera is too close to a surface)
        float max_ao_radius_cs = 0.4;
        //float max_ao_radius_cs = 100;
        ao_radius_shrinkage = min(1.0, max_ao_radius_cs / cs_ao_radius_rescale);
    }

    // Shrink the AO radius
    cs_slice_dir *= ao_radius_shrinkage;
    ao_radius *= ao_radius_shrinkage;

    float3 center_vs = ray_hit_vs.xyz;

    cs_slice_dir *= 1.0 / float(ssgi_half_sample_count);
    float2 vs_slice_dir = mul(float4(cs_slice_dir, 0, 0), frame_constants.view_constants.sample_to_view).xy;
    float3 slice_normal_vs = normalize(cross(v_vs, float3(vs_slice_dir, 0)));

    float3 proj_normal_vs = normal_vs - slice_normal_vs * dot(slice_normal_vs, normal_vs);
    float slice_contrib_weight = length(proj_normal_vs);
    proj_normal_vs /= slice_contrib_weight;

    float n_angle = fast_acos(clamp(dot(proj_normal_vs, v_vs), -1.0, 1.0)) * sign(dot(vs_slice_dir, proj_normal_vs.xy - v_vs.xy));

    float theta_cos_max1 = cos(n_angle - M_FRAC_PI_2);
    float theta_cos_max2 = cos(n_angle + M_FRAC_PI_2);

    float4 color_accum = 0.0.xxxx;

    float3 prev_sample0_vs = v_vs;
    float3 prev_sample1_vs = v_vs;

    for (uint i = 1; i <= ssgi_half_sample_count; ++i) {
        {
            float t = float(i) + rand_offset;

            float4 sample_cs = float4(ray_hit_cs.xy - cs_slice_dir * t, 0, 1);
            sample_cs.z = fetch_depth(cs_to_uv(sample_cs.xy));

            theta_cos_max1 = process_sample(i, 1, n_angle, prev_sample0_vs, sample_cs, center_vs, normal_vs, v_vs, ao_radius, theta_cos_max1, color_accum);
        }

        {
            float t = float(i) + (1.0 - rand_offset);

            float4 sample_cs = float4(ray_hit_cs.xy + cs_slice_dir * t, 0, 1);
            sample_cs.z = fetch_depth(cs_to_uv(sample_cs.xy));

            theta_cos_max2 = process_sample(i, -1, n_angle, prev_sample1_vs, sample_cs, center_vs, normal_vs, v_vs, ao_radius, theta_cos_max2, color_accum);
        }
    }

    float h1 = -fast_acos(theta_cos_max1);
    float h2 = +fast_acos(theta_cos_max2);

    float h1p = n_angle + max(h1 - n_angle, -M_FRAC_PI_2);
    float h2p = n_angle + min(h2 - n_angle, M_FRAC_PI_2);

    float inv_ao = integrate_arc(h1p, h2p, n_angle);
    col.a = max(0.0, inv_ao);
    col.rgb = color_accum.rgb;
    col *= slice_contrib_weight;

    output_tex[px] = col;
}
