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

    float2 d = uv * 2.0 - 1.0;
    float aspectRatio = float(dims.x) / float(dims.y);

    RayDesc ray;
    ray.Origin = float3(0.0, 2.0, 4.0);
    ray.Direction = normalize(float3(d.x * aspectRatio, -d.y, -1.0));
    ray.TMin = 0.001;
    ray.TMax = 100000.0;

    Payload payload;
    payload.hitValue = float3(0.0, 0.0, 0.0);

    TraceRay(acceleration_structure, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);

    output_tex[launchIndex] = float4(payload.hitValue, 1.0f);
    //output_tex[launchIndex] = float4(ray.Direction, 1.0f);
}
