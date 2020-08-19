struct ViewConstants {
    float4x4 view_to_clip;
    float4x4 clip_to_view;
    float4x4 view_to_sample;
    float4x4 sample_to_view;
    float4x4 world_to_view;
    float4x4 view_to_world;
    float2 sample_offset_pixels;
    float2 sample_offset_clip;
};

struct FrameConstants {
    ViewConstants view_constants;
    float4 mouse;
    uint frame_index;
};

[[vk::binding(0, 2)]] ConstantBuffer<FrameConstants> frame_constants;
