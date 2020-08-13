RWTexture2D<float4> input_tex;

float3 rgb_to_ycbcr(float3 col) {
    return float3(
        dot(float3(0.2126, 0.7152, 0.0722), col),
        dot(float3(-0.1146,-0.3854, 0.5), col),
        dot(float3(0.5,-0.4542,-0.0458), col)
    );
}

[numthreads(8, 8, 1)]
void main(in uint2 pix : SV_DispatchThreadID) {
    float4 rgba = input_tex[pix];
    input_tex[pix] = float4(rgb_to_ycbcr(rgba.rgb), rgba.a);
}
