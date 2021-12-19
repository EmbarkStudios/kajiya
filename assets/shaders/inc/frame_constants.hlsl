#ifndef FRAME_CONSTANTS_HLSL
#define FRAME_CONSTANTS_HLSL

#include "uv.hlsl"
#include "lights/packed.hlsl"
#include "ray_cone.hlsl"

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

struct GiCascadeConstants {
    int4 scroll_frac;
    int4 scroll_int;
    int4 voxels_scrolled_this_frame;
    float volume_size;
    float voxel_size; 
    uint2 pad;   
};

struct FrameConstants {
    ViewConstants view_constants;

    float4 sun_direction;

    uint frame_index;
    float delta_time_seconds;
    float sun_angular_radius_cos;
    uint triangle_light_count;

    float4 sun_color_multiplier;
    float4 sky_ambient;

    float world_gi_scale;
	uint pad0;
	uint pad1;
	uint pad2;

    GiCascadeConstants gi_cascades[4];
};

[[vk::binding(0, 2)]] ConstantBuffer<FrameConstants> frame_constants;

struct InstanceDynamicConstants {
    float emissive_multiplier;
};

[[vk::binding(1, 2)]] StructuredBuffer<InstanceDynamicConstants> instance_dynamic_parameters_dyn;
[[vk::binding(2, 2)]] StructuredBuffer<TriangleLightPacked> triangle_lights_dyn;

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
        return normalize(ray_dir_vs_h.xyz);
    }

    float3 ray_dir_ws() {
        return normalize(ray_dir_ws_h.xyz);
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

    // A biased position from which secondary rays can be shot without too much acne or leaking
    float3 biased_secondary_ray_origin_ws() {
        return ray_hit_ws() - ray_dir_ws() * (length(ray_hit_vs()) + length(ray_hit_ws())) * 1e-4;
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

float3 direction_world_to_view(float3 v) {
    return mul(frame_constants.view_constants.world_to_view, float4(v, 0)).xyz;
}

float3 position_world_to_view(float3 v) {
    return mul(frame_constants.view_constants.world_to_view, float4(v, 1)).xyz;
}

float3 position_world_to_clip(float3 v) {
    float4 p = mul(frame_constants.view_constants.world_to_view, float4(v, 1));
    p = mul(frame_constants.view_constants.view_to_clip, p);
    return p.xyz / p.w;
}

float3 position_world_to_sample(float3 v) {
    float4 p = mul(frame_constants.view_constants.world_to_view, float4(v, 1));
    p = mul(frame_constants.view_constants.view_to_sample, p);
    return p.xyz / p.w;
}

float pixel_cone_spread_angle_from_image_height(float image_height) {
    return atan(2.0 * frame_constants.view_constants.clip_to_view._11 / image_height);
}

RayCone pixel_ray_cone_from_image_height(float image_height) {
    RayCone res;
    res.width = 0.0;
    res.spread_angle = pixel_cone_spread_angle_from_image_height(image_height);
    return res;
}

#endif