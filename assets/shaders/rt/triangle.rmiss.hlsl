struct Payload
{
    float3 hitValue;
};

[shader("miss")]
void main(inout Payload payload : SV_RayPayload)
{
    payload.hitValue = float3(0.0, 0.1, 0.3);
}
