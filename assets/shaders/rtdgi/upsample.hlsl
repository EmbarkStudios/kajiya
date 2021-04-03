#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"
#include "../inc/gbuffer.hlsl"

#define USE_SSAO_STEERING 1

[[vk::binding(0)]] Texture2D<float4> ssgi_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(3)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(4)]] Texture2D<float> half_depth_tex;
[[vk::binding(5)]] Texture2D<float4> ssao_tex;
[[vk::binding(6)]] RWTexture2D<float4> output_tex;
[[vk::binding(7)]] cbuffer _ {
    float4 output_tex_size;
    int4 spatial_resolve_offsets[16 * 4 * 8];
};

float4 process_sample(float2 soffset, float4 ssgi, float depth, float3 normal, float center_depth, float3 center_normal, inout float w_sum) {
    if (depth != 0.0)
    {
        float depth_diff = 1.0 - (center_depth / depth);
        float depth_factor = exp2(-200.0 * abs(depth_diff));

        float normal_factor = max(0.0, dot(normal, center_normal));
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;

        float w = 1;
        w *= depth_factor;  // TODO: differentials
        w *= normal_factor;
        //w *= exp(-dot(soffset, soffset));

        w_sum += w;
        return ssgi * w;
    } else {
        return 0.0.xxxx;
    }
}

static float ggx_ndf_unnorm(float a2, float cos_theta) {
	float denom_sqrt = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
	return a2 / (denom_sqrt * denom_sqrt);
}


[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float4 result = 0.0.xxxx;
    float ex = 0.0;
    float ex2 = 0.0;
    float w_sum = 0.0;
    float w_sum2 = 0.0;

    float center_depth = depth_tex[px];
    if (center_depth != 0.0) {
        float3 center_normal_vs = mul(frame_constants.view_constants.world_to_view, float4(unpack_normal_11_10_11(gbuffer_tex[px].y), 0)).xyz;
        const float center_ssao = ssao_tex[px].a;
        const float rel_std_dev = ssgi_tex[px].a;

        const float2 uv = get_uv(px, output_tex_size);
        const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, center_depth);

        const float world_space_kernel_radius = 0.5 * frame_constants.world_gi_scale;
        const float filter_radius_ss = center_ssao * world_space_kernel_radius * frame_constants.view_constants.view_to_clip[1][1] / -view_ray_context.ray_hit_vs().z;

        w_sum = 0.0;
        result = 0.0.xxxx;

        ex = 0;
        ex2 = 0;
        w_sum2 = 0.0;

        //const int sample_count = int(lerp(6, 16, saturate(rel_std_dev)));
        const int sample_count = 8;
        //const int sample_count = 1;
        const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + frame_constants.frame_index) & 3;

        const uint filter_idx = uint(clamp(filter_radius_ss * 7.0, 0.0, 7.0));

        // TODO: not using sample 0 removes pixellation, but potentially loses small detail
        for (uint sample_i = 1; sample_i < sample_count + 1; ++sample_i) {

            // Swizzle as .yx to avoid using the same samples as the previous filter
            const int2 sample_offset = spatial_resolve_offsets[(px_idx_in_quad * 16 + sample_i) + 64 * filter_idx].yx;

            int2 sample_px = px / 2 + sample_offset;

            float3 sample_normal_vs = half_view_normal_tex[sample_px].rgb;
            float sample_depth = half_depth_tex[sample_px];
            float4 sample_val = ssgi_tex[sample_px];
            float sample_ssao = ssao_tex[sample_px * 2].a;

            if (sample_depth != 0) {
                float wt = 1;

                //wt *= exp2(-spatial_sharpness * sqrt(float(dot(sample_offset, sample_offset))));
                //wt *= pow(saturate(dot(center_normal_vs, sample_normal_vs)), 100);
                wt *= ggx_ndf_unnorm(0.01, saturate(dot(center_normal_vs, sample_normal_vs)));
                wt *= exp2(-200.0 * abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)));

                #if USE_SSAO_STEERING
                    wt *= exp2(-20.0 * abs(sample_ssao - center_ssao));
                #endif

                result += sample_val * wt;
                w_sum += wt;

                /*wt = 1;
                //wt *= pow(saturate(dot(center_normal_vs, sample_normal_vs)), 30);
                wt *= ggx_ndf_unnorm(0.01, saturate(dot(center_normal_vs, sample_normal_vs)));
                wt *= exp2(-200.0 * abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)));*/

                const float luma = calculate_luma(sample_val.rgb);
                ex += luma * wt;
                ex2 += luma * luma * wt;
                w_sum2 += wt;
            }
        }

        #if 0
            // Output normal for debug purposes
            result = float4(saturate(normalize(center_normal_vs) * 0.5 + 0.5), 1);
            result *= result;
            w_sum = 1;
        #endif
    } else {
        result = 0.0.xxxx;
    }

#if 0
    result = saturate(1 - 1e-2 / (center_depth + 1e-5));
    w_sum = 1;
#endif

    float dev = 1;
    if (w_sum2 > 1e-200) {
        ex /= w_sum2;
        ex2 /= w_sum2;
        dev = sqrt(abs(ex * ex - ex2));
    }

    if (w_sum > 1e-200) {
        //output_tex[px] = result / w_sum;

        output_tex[px] = float4(result.rgb / w_sum, dev);
    } else {
        //output_tex[px] = ssgi_tex[px / 2];
        output_tex[px] = float4(ssgi_tex[px / 2].rgb, dev);
    }
}
