#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"

struct Payload {
    float3 hitValue;
};

struct Attribute {
    float2 bary;
};

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] RWTexture2D<float4> output_tex;

[shader("raygeneration")]
void main()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims = DispatchRaysDimensions().xy;

    float2 pixelCenter = launchIndex + 0.5;
    float2 uv = pixelCenter / dims.xy;

    ViewConstants view_constants = frame_constants.view_constants;
    float4 ray_dir_cs = float4(uv_to_cs(uv), 0.0, 1.0);
    float4 ray_dir_vs = mul(view_constants.sample_to_view, ray_dir_cs);
    float4 ray_dir_ws = mul(view_constants.view_to_world, ray_dir_vs);

    float4 ray_origin_cs = float4(uv_to_cs(uv), 1.0, 1.0);
    float4 ray_origin_ws = mul(view_constants.view_to_world, mul(view_constants.sample_to_view, ray_origin_cs));
    ray_origin_ws /= ray_origin_ws.w;

    /*float2 d = uv * 2.0 - 1.0;
    float aspectRatio = float(dims.x) / float(dims.y);*/

    RayDesc ray;
    ray.Origin = ray_origin_ws.xyz;
    ray.Direction = normalize(ray_dir_ws.xyz);
    ray.TMin = 0.001;
    ray.TMax = 100000.0;

    Payload payload;
    payload.hitValue = 0.0.xxx;

    TraceRay(acceleration_structure, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);

    if (launchIndex.x > 1280 / 2 && any(payload.hitValue > 0.0.xxx)) {
        output_tex[launchIndex] = float4(payload.hitValue, 1.0f);
        //output_tex[launchIndex] = float4(ray.Direction, 1.0f);
        //output_tex[launchIndex] = float4(payload.hitValue.xxx, 1.0f);
    }
}
