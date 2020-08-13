struct Payload
{
    float3 hitValue;
};

struct Attribute
{
    float2 bary;
};

RaytracingAccelerationStructure g_topLevel : register(t0, space0);
RWTexture2D<float4> g_output : register(u1, space0);

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

    TraceRay(g_topLevel, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);

    g_output[launchIndex] = float4(payload.hitValue, 1.0f);
    //g_output[launchIndex] = float4(ray.Direction, 1.0f);
}
