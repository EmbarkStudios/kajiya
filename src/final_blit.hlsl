[[vk::binding(0)]] Texture2D main_tex;
[[vk::binding(1)]] Texture2D gui_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;
//[[vk::binding(3)]] SamplerState sampler_lnc;

[[vk::push_constant]]
struct {
    float2 main_tex_size;
} push_constants;

float linear_to_srgb(float v) {
    if (v <= 0.0031308) {
        return v * 12.92;
    } else {
        return pow(v, (1.0/2.4)) * (1.055) - 0.055;
    }
}

float3 linear_to_srgb(float3 v) {
	return float3(
		linear_to_srgb(v.x), 
		linear_to_srgb(v.y), 
		linear_to_srgb(v.z));
}

[numthreads(8, 8, 1)]
void main(in uint2 dispatch_id : SV_DispatchThreadID) {
    #if 1
    /*float4 main = main_tex.SampleLevel(
        sampler_lnc,
        float2(dispatch_id + 0.5) * push_constants.main_tex_size,
        0);*/

    // Sampler temporarily disabled because push descriptors aren't usable with immutable samplers
    float4 main = main_tex[dispatch_id];

    //float4 gui = gui_tex.Load(uint3(dispatch_id, 0));
    //float4 gui = float4(0.7, 0.4, 0.1, 0.5);
    float4 result = main;
    result.rgb = linear_to_srgb(clamp(result.rgb, 0.0, 1.0));
    //result.rgb = result.rgb * (1.0 - gui.a) + gui.rgb;
    #else
    float4 result = float4(0.7, 0.4, 0.1, 1.0);
    #endif
    output_tex[dispatch_id] = result;
}
