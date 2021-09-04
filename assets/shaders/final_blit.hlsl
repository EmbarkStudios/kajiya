[[vk::binding(0)]] Texture2D<float4> main_tex;
[[vk::binding(1)]] Texture2D<float4> gui_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;
[[vk::binding(3)]] cbuffer _ {
    float4 main_tex_size;
    float4 output_tex_size;
};

#include "inc/image.hlsl"

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

struct LinearToSrgbRemap {
    static LinearToSrgbRemap create() {
        LinearToSrgbRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return float4(linear_to_srgb(v.rgb), 1.0);
    }
};

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    #if 1
    float3 main;
    if (any(main_tex_size.xy != output_tex_size.xy)) {
        main = image_sample_catmull_rom(
            TextureImage::from_parts(main_tex, main_tex_size.xy),
            (px + 0.5) / output_tex_size.xy,
            LinearToSrgbRemap::create()
        ).rgb;
    } else {
        main = linear_to_srgb(saturate(main_tex[px].rgb));
    }
    float4 gui = gui_tex[px];

    float3 result = main.rgb * (1.0 - gui.a) + gui.rgb;
    //float3 result = lerp(main, gui.rgb, gui.a);
    #else
    float3 result = float3(0.7, 0.4, 0.1);
    #endif

    output_tex[px] = float4(result, 1);
}
