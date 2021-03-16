[[vk::binding(0)]] Texture2D<float3> input_tex;
[[vk::binding(1)]] RWTexture2D<float2> output_tex;

[numthreads(8, 8, 1)]
void main(int2 px: SV_DispatchThreadID) {
	float3 col = input_tex[px].rgb;
    //float luminance = clamp(dot(float3(0.2126, 0.7152, 0.0722), col), 1e-8, 1e8);
    float luminance = clamp(dot(float3(0.2126, 0.7152, 0.0722), col), 0.05, 1e8);
    //float luminance = clamp(dot(float3(0.2126, 0.7152, 0.0722), col), 0.3, 1e8);
	output_tex[px] = float2(log(luminance), luminance);
}
