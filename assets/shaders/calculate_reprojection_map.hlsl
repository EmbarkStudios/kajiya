#include "inc/frame_constants.hlsl"
#include "inc/pack_unpack.hlsl"
#include "inc/uv.hlsl"
#include "inc/samplers.hlsl"
#include "inc/bilinear.hlsl"

[[vk::binding(0)]] Texture2D<float> depth_tex;
[[vk::binding(1)]] Texture2D<float3> geometric_normal_tex;
[[vk::binding(2)]] Texture2D<float> prev_depth_tex;
[[vk::binding(3)]] Texture2D<float3> velocity_tex;
[[vk::binding(4)]] RWTexture2D<float4> output_tex;
[[vk::binding(5)]] cbuffer _ {
    float4 output_tex_size;
};


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);

    if (depth_tex[px] == 0.0) {
        float4 pos_cs = float4(uv_to_cs(uv), 0.0, 1.0);
        float4 pos_vs = mul(frame_constants.view_constants.clip_to_view, pos_cs);

        float4 prev_vs = pos_vs;
        
        float4 prev_cs = mul(frame_constants.view_constants.view_to_clip, prev_vs);
        float4 prev_pcs = mul(frame_constants.view_constants.clip_to_prev_clip, prev_cs);

        float2 prev_uv = cs_to_uv(prev_pcs.xy);
        float2 uv_diff = prev_uv - uv;

        output_tex[px] = float4(uv_diff, 0, 0);
        return;
    }


    float3 eye_pos = mul(frame_constants.view_constants.view_to_world, float4(0, 0, 0, 1)).xyz;

    float depth = 0.0;
    {
        const int k = 0;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float s_depth = depth_tex[px + int2(x, y)];
                if (s_depth != 0.0) {
                    depth = max(depth, s_depth);
                }
            }
        }
    }

    float3 normal_vs = geometric_normal_tex[px] * 2.0 - 1.0;

    float4 pos_cs = float4(uv_to_cs(uv), depth, 1.0);
    float4 pos_vs = mul(frame_constants.view_constants.clip_to_view, pos_cs);
    float dist_to_point = -(pos_vs.z / pos_vs.w);

    float4 prev_vs = pos_vs / pos_vs.w;
    prev_vs.xyz += float4(velocity_tex[px].xyz, 0).xyz;
    
    //float4 prev_cs = mul(frame_constants.view_constants.prev_view_to_prev_clip, prev_vs);
    float4 prev_cs = mul(frame_constants.view_constants.view_to_clip, prev_vs);
    float4 prev_pcs = mul(frame_constants.view_constants.clip_to_prev_clip, prev_cs);

    float2 prev_uv = cs_to_uv(prev_pcs.xy / prev_pcs.w);
    float2 uv_diff = prev_uv - uv;

    // Account for quantization of the `uv_diff` in R16G16B16A16_SNORM.
    // This is so we calculate validity masks for pixels that the users will actually be using.
    uv_diff = floor(uv_diff * 32767.0 + 0.5) / 32767.0;
    prev_uv = uv + uv_diff;

    float4 prev_pvs = mul(frame_constants.view_constants.prev_clip_to_prev_view, prev_pcs);
    prev_pvs /= prev_pvs.w;

    // Based on "Fast Denoising with Self Stabilizing Recurrent Blurs"
    
    float plane_dist_prev = dot(normal_vs, prev_pvs.xyz);

    // Note: departure from the quoted technique: they calculate reprojected sample depth by linearly
    // scaling plane distance with view-space Z, which is not correct unless the plane is aligned with view.
    // Instead, the amount that distance actually increases with depth is simply `normal_vs.z`.

    // Note: bias the minimum distance increase, so that reprojection at grazing angles has a sharper cutoff.
    // This can introduce shimmering a grazing angles, but also reduces reprojection artifacts on surfaces
    // which flip their normal from back- to fron-facing across a frame. Such reprojection smears a few
    // pixels along a wide area, creating a glitchy look.
    float plane_dist_prev_dz = min(-0.2, normal_vs.z);
    //float plane_dist_prev_dz = -normal_vs.z;

    const Bilinear bilinear_at_prev = get_bilinear_filter(prev_uv, output_tex_size.xy);
    float2 prev_gather_uv = (bilinear_at_prev.origin + 1.0) / output_tex_size.xy;
    float4 prev_depth = prev_depth_tex.GatherRed(sampler_nnc, prev_gather_uv).wzxy;

    float4 prev_view_z = rcp(prev_depth * -frame_constants.view_constants.prev_clip_to_prev_view._43);

    // Note: departure from the quoted technique: linear offset from zero distance at previous position instead of scaling.
    float4 quad_dists = abs(plane_dist_prev_dz * (prev_view_z - prev_pvs.z));

    // TODO: reject based on normal too? Potentially tricky under rotations.

    // Resolution-dependent. Was tweaked for 1080p
    const float acceptance_threshold = 0.001 * (1080 / output_tex_size.y);

    // Reduce strictness at grazing angles, where distances grow due to perspective
    const float3 pos_vs_norm = normalize(pos_vs.xyz / pos_vs.w);
    const float ndotv = dot(normal_vs, pos_vs_norm);

    float4 quad_validity = step(quad_dists, acceptance_threshold * dist_to_point / -ndotv);

    quad_validity.x *= all(bilinear_at_prev.px0() >= 0) && all(bilinear_at_prev.px0() < uint2(output_tex_size.xy));
    quad_validity.y *= all(bilinear_at_prev.px1() >= 0) && all(bilinear_at_prev.px1() < uint2(output_tex_size.xy));
    quad_validity.z *= all(bilinear_at_prev.px2() >= 0) && all(bilinear_at_prev.px2() < uint2(output_tex_size.xy));
    quad_validity.w *= all(bilinear_at_prev.px3() >= 0) && all(bilinear_at_prev.px3() < uint2(output_tex_size.xy));

    float validity = dot(quad_validity, float4(1, 2, 4, 8)) / 15.0;

    float2 texel_center_offset = abs(0.5 - frac(prev_uv * output_tex_size.xy));
    float accuracy = 1.0 - texel_center_offset.x - texel_center_offset.y;

    output_tex[px] = float4(
        uv_diff,
        validity,
        accuracy
    );
}
