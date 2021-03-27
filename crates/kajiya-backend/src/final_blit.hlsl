[[vk::binding(0)]] Texture2D<float4> main_tex;
[[vk::binding(1)]] Texture2D<float4> gui_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;

[[vk::push_constant]]
struct {
    float2 main_tex_size;
    float2 output_tex_size;
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

float3 fetch_main_px(int2 px) {
	return linear_to_srgb(saturate(main_tex[px].rgb));
}

float3 cubic_hermite(float3 A, float3 B, float3 C, float3 D, float t) {
	float t2 = t*t;
    float t3 = t*t*t;
    float3 a = -A/2.0 + (3.0*B)/2.0 - (3.0*C)/2.0 + D/2.0;
    float3 b = A - (5.0*B)/2.0 + 2.0*C - D / 2.0;
    float3 c = -A/2.0 + C/2.0;
   	float3 d = B;
    
    return a*t3 + b*t2 + c*t + d;
}

// https://www.shadertoy.com/view/MllSzX
float3 fetch_main_catmull_rom(float2 P) {
    float2 pixel = P * push_constants.main_tex_size.xy + 0.5;
    float2 c_onePixel = 1.0 / push_constants.main_tex_size;
    float2 c_twoPixels = 2.0 / push_constants.main_tex_size;
    
    float2 frc = frac(pixel);
    //pixel = floor(pixel) / main_tex_size.xy - float2(c_onePixel/2.0);
    int2 ipixel = int2(pixel) - 1;
    
    float3 C00 = fetch_main_px(ipixel + int2(-1 ,-1));
    float3 C10 = fetch_main_px(ipixel + int2( 0        ,-1));
    float3 C20 = fetch_main_px(ipixel + int2( 1 ,-1));
    float3 C30 = fetch_main_px(ipixel + int2( 2,-1));
    
    float3 C01 = fetch_main_px(ipixel + int2(-1 , 0));
    float3 C11 = fetch_main_px(ipixel + int2( 0        , 0));
    float3 C21 = fetch_main_px(ipixel + int2( 1 , 0));
    float3 C31 = fetch_main_px(ipixel + int2( 2, 0));    
    
    float3 C02 = fetch_main_px(ipixel + int2(-1 , 1));
    float3 C12 = fetch_main_px(ipixel + int2( 0        , 1));
    float3 C22 = fetch_main_px(ipixel + int2( 1 , 1));
    float3 C32 = fetch_main_px(ipixel + int2( 2, 1));    
    
    float3 C03 = fetch_main_px(ipixel + int2(-1 , 2));
    float3 C13 = fetch_main_px(ipixel + int2( 0        , 2));
    float3 C23 = fetch_main_px(ipixel + int2( 1 , 2));
    float3 C33 = fetch_main_px(ipixel + int2( 2, 2));    
    
    float3 CP0X = cubic_hermite(C00, C10, C20, C30, frc.x);
    float3 CP1X = cubic_hermite(C01, C11, C21, C31, frc.x);
    float3 CP2X = cubic_hermite(C02, C12, C22, C32, frc.x);
    float3 CP3X = cubic_hermite(C03, C13, C23, C33, frc.x);
    
    return cubic_hermite(CP0X, CP1X, CP2X, CP3X, frc.y);
}

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    #if 1
    float3 main;
    if (any(push_constants.main_tex_size != push_constants.output_tex_size)) {
        main = fetch_main_catmull_rom((px + 0.5) / push_constants.output_tex_size);
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
