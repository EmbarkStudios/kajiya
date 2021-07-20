#ifndef FRAME_CONSTANTS_HLSL
#define FRAME_CONSTANTS_HLSL

#include "uv.hlsl"

struct ViewConstants {
    float4x4 view_to_clip;
    float4x4 clip_to_view;
    float4x4 view_to_sample;
    float4x4 sample_to_view;
    float4x4 world_to_view;
    float4x4 view_to_world;

    float4x4 clip_to_prev_clip;

    float4x4 prev_view_to_prev_clip;
    float4x4 prev_clip_to_prev_view;
    float4x4 prev_world_to_prev_view;
    float4x4 prev_view_to_prev_world;

    float2 sample_offset_pixels;
    float2 sample_offset_clip;
};

struct FrameConstants {
    ViewConstants view_constants;

    float4 sun_direction;

    float sun_angular_radius_cos;
    uint frame_index;
    float world_gi_scale;
    float global_fog_thickness;

    float4 sun_color_multiplier;
    float4 sky_ambient;

    float delta_time_seconds;
};

[[vk::binding(0, 2)]] ConstantBuffer<FrameConstants> frame_constants;

struct InstanceDynamicConstants {
    float emissive_multiplier;
};

[[vk::binding(1, 2)]] StructuredBuffer<InstanceDynamicConstants> instance_dynamic_constants_dyn;

struct ViewRayContext {
    float4 ray_dir_cs;
    float4 ray_dir_vs_h;
    float4 ray_dir_ws_h;

    float4 ray_origin_cs;
    float4 ray_origin_vs_h;
    float4 ray_origin_ws_h;

    float4 ray_hit_cs;
    float4 ray_hit_vs_h;
    float4 ray_hit_ws_h;

    float3 ray_dir_vs() {
        return ray_dir_vs_h.xyz;
    }

    float3 ray_dir_ws() {
        return ray_dir_ws_h.xyz;
    }

    float3 ray_origin_vs() {
        return ray_origin_vs_h.xyz / ray_origin_vs_h.w;
    }

    float3 ray_origin_ws() {
        return ray_origin_ws_h.xyz / ray_origin_ws_h.w;
    }

    float3 ray_hit_vs() {
        return ray_hit_vs_h.xyz / ray_hit_vs_h.w;
    }

    float3 ray_hit_ws() {
        return ray_hit_ws_h.xyz / ray_hit_ws_h.w;
    }

    static ViewRayContext from_uv(float2 uv) {
        ViewConstants view_constants = frame_constants.view_constants;

        ViewRayContext res;
        res.ray_dir_cs = float4(uv_to_cs(uv), 0.0, 1.0);
        res.ray_dir_vs_h = mul(view_constants.sample_to_view, res.ray_dir_cs);
        res.ray_dir_ws_h = mul(view_constants.view_to_world, res.ray_dir_vs_h);

        res.ray_origin_cs = float4(uv_to_cs(uv), 1.0, 1.0);
        res.ray_origin_vs_h = mul(view_constants.sample_to_view, res.ray_origin_cs);
        res.ray_origin_ws_h = mul(view_constants.view_to_world, res.ray_origin_vs_h);

        return res;
    }

    static ViewRayContext from_uv_and_depth(float2 uv, float depth) {
        ViewConstants view_constants = frame_constants.view_constants;

        ViewRayContext res;
        res.ray_dir_cs = float4(uv_to_cs(uv), 0.0, 1.0);
        res.ray_dir_vs_h = mul(view_constants.sample_to_view, res.ray_dir_cs);
        res.ray_dir_ws_h = mul(view_constants.view_to_world, res.ray_dir_vs_h);

        res.ray_origin_cs = float4(uv_to_cs(uv), 1.0, 1.0);
        res.ray_origin_vs_h = mul(view_constants.sample_to_view, res.ray_origin_cs);
        res.ray_origin_ws_h = mul(view_constants.view_to_world, res.ray_origin_vs_h);

        res.ray_hit_cs = float4(uv_to_cs(uv), depth, 1.0);
        res.ray_hit_vs_h = mul(view_constants.sample_to_view, res.ray_hit_cs);
        res.ray_hit_ws_h = mul(view_constants.view_to_world, res.ray_hit_vs_h);

        return res;
    }
};

float3 get_eye_position() {
    float4 eye_pos_h = mul(frame_constants.view_constants.view_to_world, float4(0, 0, 0, 1));
    return eye_pos_h.xyz / eye_pos_h.w;
}

float depth_to_view_z(float depth) {
    return rcp(depth * -frame_constants.view_constants.clip_to_view._43);
}

float3 direction_view_to_world(float3 v) {
    return mul(frame_constants.view_constants.view_to_world, float4(v, 0)).xyz;
}

float3 world_to_view(float3 v) {
    return mul(frame_constants.view_constants.world_to_view, float4(v, 0)).xyz;
}

#endif