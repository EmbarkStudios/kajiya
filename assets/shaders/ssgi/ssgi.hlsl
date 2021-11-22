#include "../inc/math_const.hlsl"
#include "../inc/math.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/gbuffer.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> half_depth_tex;
[[vk::binding(2)]] Texture2D<float4> view_normal_tex;
[[vk::binding(3)]] Texture2D<float4> prev_radiance_tex;
[[vk::binding(4)]] Texture2D<float4> reprojection_tex;
[[vk::binding(5)]] RWTexture2D<float4> output_tex;
//[[vk::binding(6)]] RWTexture2D<float4> bent_normal_out_tex;

#define USE_AO_ONLY 1

[[vk::binding(6)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

#if 1
    // Micro-occlusion settings used for denoising

    static const uint SSGI_HALF_SAMPLE_COUNT = 6;
    #define SSGI_KERNEL_RADIUS (50.0 * output_tex_size.w)
    #define MAX_KERNEL_RADIUS_CS 0.4
    #define USE_KERNEL_DISTANCE_SCALING 0
    #define USE_RANDOM_JITTER 0
#else
    // Crazy settings for testing with the Cornell Box

    static const uint SSGI_HALF_SAMPLE_COUNT = 32;
    #define SSGI_KERNEL_RADIUS 5
    #define MAX_KERNEL_RADIUS_CS 100.0
    #define USE_KERNEL_DISTANCE_SCALING 1
    #define USE_RANDOM_JITTER 1
#endif

static const float temporal_rotations[] = { 60.0, 300.0, 180.0, 240.0, 120.0, 0.0 };
static const float temporal_offsets[] = { 0.0, 0.5, 0.25, 0.75 };

// [Drobot2014a] Low Level Optimizations for GCN
float fast_sqrt(float x) {
    return asfloat(0x1fbd1df5 + (asuint(x) >> 1u));
}

// [Eberly2014] GPGPU Programming for Games and Science
float fast_acos(float inX) { 
    float x = abs(inX); 
    float res = -0.156583f * x + M_FRAC_PI_2; 
    res *= fast_sqrt(1.0f - x); 
    return (inX >= 0) ? res : M_PI - res;
}

struct Ray {
	float3 o;
	float3 d;
};

float3 fetch_lighting(float2 uv) {
    //return 0.0.xxx;
    int2 px = int2(input_tex_size.xy * uv);
    float4 reproj = reprojection_tex[px];
    return lerp(0.0, prev_radiance_tex[int2(input_tex_size.xy * (uv + reproj.xy))].xyz, reproj.z);
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

float update_horizion_angle(float prev, float cur, float blend) {
    return cur > prev ? lerp(prev, cur, blend) : prev;
}

float intersect_dir_plane_onesided(float3 dir, float3 normal, float3 pt) {
    float d = -dot(pt, normal);
    float t = d / max(1e-5, -dot(dir, normal));
    return t;
}

float3 project_point_on_plane(float3 pt, float3 normal) {
    return pt - normal * dot(pt, normal);
}

float process_sample(uint i, float intsgn, float n_angle, inout float3 prev_sample_vs, float4 sample_cs, float3 center_vs, float3 normal_vs, float3 v_vs, float kernel_radius_vs, float theta_cos_max, inout float4 color_accum) {
    if (sample_cs.z > 0) {
        float4 sample_vs4 = mul(frame_constants.view_constants.sample_to_view, sample_cs);
        float3 sample_vs = sample_vs4.xyz / sample_vs4.w;
        float3 sample_vs_offset = sample_vs - center_vs;
        float sample_vs_offset_len = length(sample_vs_offset);

        float sample_theta_cos = dot(sample_vs_offset, v_vs) / sample_vs_offset_len;
        const float sample_distance_normalized = sample_vs_offset_len / kernel_radius_vs;

        if (sample_distance_normalized < 1.0) {
            //const float sample_influence = 1;
            const float sample_influence = 1.0 - sample_distance_normalized * sample_distance_normalized;

            bool sample_visible = sample_theta_cos >= theta_cos_max;
            float theta_cos_prev = theta_cos_max;
            float theta_delta = theta_cos_max;
            theta_cos_max = update_horizion_angle(theta_cos_max, sample_theta_cos, sample_influence);
            theta_delta = theta_cos_max - theta_delta;

            if (sample_visible) {
                float3 lighting = fetch_lighting(cs_to_uv(sample_cs.xy));

                float3 sample_normal_vs = fetch_normal_vs(cs_to_uv(sample_cs.xy));
                float theta_cos_prev_trunc = theta_cos_prev;

#if 1
                if (i > 0) {
                    // Account for the sampled surface's normal, and how it's facing the center pixel

                    float3 p1 = prev_sample_vs * min(
                        intersect_dir_plane_onesided(prev_sample_vs, sample_normal_vs, sample_vs),
                        intersect_dir_plane_onesided(prev_sample_vs, normal_vs, center_vs)
                    );

                    theta_cos_prev_trunc = clamp(dot(normalize(p1 - center_vs), v_vs), theta_cos_prev_trunc, theta_cos_max);
                }
#endif

                {
                    // Scale the lighting contribution by the cosine factor

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
                }

                color_accum += float4(lighting, 1.0);
            }
        }

        prev_sample_vs = sample_vs;
    } else {
        // Sky; assume no occlusion
        theta_cos_max = update_horizion_angle(theta_cos_max, -1, 1);
    }

    return theta_cos_max;
}

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);

    const float depth = half_depth_tex[px];
    if (0.0 == depth) {
        output_tex[px] = float4(0, 0, 0, 1);
        return;
    }

    float4 gbuffer_packed = gbuffer_tex[px * 2];

    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    const float3 normal_vs = normalize(mul(frame_constants.view_constants.world_to_view, float4(gbuffer.normal, 0)).xyz);

    float4 col = 0.0.xxxx;
    float kernel_radius_cs = SSGI_KERNEL_RADIUS;

    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);
    float3 v_vs = -normalize(view_ray_context.ray_dir_vs());

    float4 ray_hit_cs = view_ray_context.ray_hit_cs;
    float3 ray_hit_vs = view_ray_context.ray_hit_vs();
    
    float spatial_direction_noise = 1.0 / 16.0 * ((((px.x + px.y) & 3) << 2) + (px.x & 3));
    float temporal_direction_noise = temporal_rotations[frame_constants.frame_index % 6] / 360.0;
    float spatial_offset_noise = (1.0 / 4.0) * ((px.y - px.x) & 3);
    float temporal_offset_noise = temporal_offsets[frame_constants.frame_index / 6 % 4];

#if USE_RANDOM_JITTER
    uint seed0 = hash3(uint3(frame_constants.frame_index, px.x, px.y));
    spatial_direction_noise += uint_to_u01_float(seed0) * 0.1;
#endif

    float ss_angle = frac(spatial_direction_noise + temporal_direction_noise) * M_PI;
    float rand_offset = frac(spatial_offset_noise + temporal_offset_noise);

    float2 cs_slice_dir = float2(cos(ss_angle) * input_tex_size.y / input_tex_size.x, sin(ss_angle));

    float kernel_radius_shrinkage;
    {
        // Convert AO radius into world scale
        #if USE_KERNEL_DISTANCE_SCALING
            const float cs_kernel_radius_scaled = kernel_radius_cs * frame_constants.view_constants.view_to_clip[1][1] / -ray_hit_vs.z;
        #else
            const float cs_kernel_radius_scaled = kernel_radius_cs;
        #endif

        cs_slice_dir *= cs_kernel_radius_scaled;

        // Calculate AO radius shrinkage (if camera is too close to a surface)
        float max_kernel_radius_cs = MAX_KERNEL_RADIUS_CS;

        //float max_kernel_radius_cs = 100;
        kernel_radius_shrinkage = min(1.0, max_kernel_radius_cs / cs_kernel_radius_scaled);
    }

    // Shrink the AO radius
    cs_slice_dir *= kernel_radius_shrinkage;

    const float kernel_radius_vs = kernel_radius_cs * kernel_radius_shrinkage * -ray_hit_vs.z;

    float3 center_vs = ray_hit_vs.xyz;

    cs_slice_dir *= 1.0 / float(SSGI_HALF_SAMPLE_COUNT);
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

    int2 prev_sample_coord0 = px;
    int2 prev_sample_coord1 = px;

    for (uint i = 0; i < SSGI_HALF_SAMPLE_COUNT; ++i) {
        {
            float t = float(i) + rand_offset;

            float4 sample_cs = float4(ray_hit_cs.xy - cs_slice_dir * t, 0, 1);
            int2 sample_px = int2(output_tex_size.xy * cs_to_uv(sample_cs.xy));

            [flatten] if (any(sample_px != prev_sample_coord0)) {
                prev_sample_coord0 = sample_px;
                sample_cs.z = half_depth_tex[sample_px];
                theta_cos_max1 = process_sample(i, 1, n_angle, prev_sample0_vs, sample_cs, center_vs, normal_vs, v_vs, kernel_radius_vs, theta_cos_max1, color_accum);
            }
        }

        {
            float t = float(i) + (1.0 - rand_offset);

            float4 sample_cs = float4(ray_hit_cs.xy + cs_slice_dir * t, 0, 1);
            int2 sample_px = int2(output_tex_size.xy * cs_to_uv(sample_cs.xy));

            [flatten] if (any(sample_px != prev_sample_coord1)) {
                prev_sample_coord1 = sample_px;
                sample_cs.z = half_depth_tex[sample_px];
                theta_cos_max2 = process_sample(i, -1, n_angle, prev_sample1_vs, sample_cs, center_vs, normal_vs, v_vs, kernel_radius_vs, theta_cos_max2, color_accum);
            }
        }
    }

    float h1 = -fast_acos(theta_cos_max1);
    float h2 = +fast_acos(theta_cos_max2);

    float h1p = n_angle + max(h1 - n_angle, -M_FRAC_PI_2);
    float h2p = n_angle + min(h2 - n_angle, M_FRAC_PI_2);

    float inv_ao = integrate_arc(h1p, h2p, n_angle);
    col.a = max(0.0, inv_ao);
    
    #if USE_AO_ONLY
        col.rgb = col.a;
    #else
        col.rgb = color_accum.rgb;
    #endif

    col *= slice_contrib_weight;

    /*float bent_normal_angle = h1p + h2p - n_angle * 2;
    float3 bent_normal_dir = sin(bent_normal_angle) * cross(slice_normal_vs, normal_vs) + cos(bent_normal_angle) * normal_vs;
    bent_normal_dir = bent_normal_dir;*/

    output_tex[px] = max(0.0, col);
    //bent_normal_out_tex[px] = float4(bent_normal_dir, 0);// / slice_contrib_weight;
}
