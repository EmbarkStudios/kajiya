#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> reprojection_tex;
[[vk::binding(3)]] RWTexture2D<float4> output_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 output_tex_size;
    float2 jitter;
};

#define ENCODING_VARIANT 2

float3 decode_rgb(float3 a) {
    #if 0 == ENCODING_VARIANT
    return a;
    #elif 1 == ENCODING_VARIANT
    return sqrt(a);
    #elif 2 == ENCODING_VARIANT
    return log(1+sqrt(a));
    #endif
}

float3 encode_rgb(float3 a) {
    #if 0 == ENCODING_VARIANT
    return a;
    #elif 1 == ENCODING_VARIANT
    return a * a;
    #elif 2 == ENCODING_VARIANT
    a = exp(a) - 1;
    return a * a;
    #endif
}

float3 decode(float3 a) {
    return decode_rgb(a);
}

float3 encode(float3 a) {
    return encode_rgb(a);
}

float3 fetch_history(float2 uv) {
	return decode(
        history_tex.SampleLevel(sampler_lnc, uv, 0).xyz
    );
}

float3 fetch_history_px(int2 px) {
	return decode(history_tex[px].xyz);
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
float3 fetch_history_catmull_rom(float2 P) {
    float2 pixel = P * output_tex_size.xy + 0.5;
    float2 c_onePixel = output_tex_size.zw;
    float2 c_twoPixels = output_tex_size.zw * 2.0;
    
    float2 frc = frac(pixel);
    //pixel = floor(pixel) / output_tex_size.xy - float2(c_onePixel/2.0);
    int2 ipixel = int2(pixel) - 1;
    
    float3 C00 = fetch_history_px(ipixel + int2(-1 ,-1));
    float3 C10 = fetch_history_px(ipixel + int2( 0        ,-1));
    float3 C20 = fetch_history_px(ipixel + int2( 1 ,-1));
    float3 C30 = fetch_history_px(ipixel + int2( 2,-1));
    
    float3 C01 = fetch_history_px(ipixel + int2(-1 , 0));
    float3 C11 = fetch_history_px(ipixel + int2( 0        , 0));
    float3 C21 = fetch_history_px(ipixel + int2( 1 , 0));
    float3 C31 = fetch_history_px(ipixel + int2( 2, 0));    
    
    float3 C02 = fetch_history_px(ipixel + int2(-1 , 1));
    float3 C12 = fetch_history_px(ipixel + int2( 0        , 1));
    float3 C22 = fetch_history_px(ipixel + int2( 1 , 1));
    float3 C32 = fetch_history_px(ipixel + int2( 2, 1));    
    
    float3 C03 = fetch_history_px(ipixel + int2(-1 , 2));
    float3 C13 = fetch_history_px(ipixel + int2( 0        , 2));
    float3 C23 = fetch_history_px(ipixel + int2( 1 , 2));
    float3 C33 = fetch_history_px(ipixel + int2( 2, 2));    
    
    float3 CP0X = cubic_hermite(C00, C10, C20, C30, frc.x);
    float3 CP1X = cubic_hermite(C01, C11, C21, C31, frc.x);
    float3 CP2X = cubic_hermite(C02, C12, C22, C32, frc.x);
    float3 CP3X = cubic_hermite(C03, C13, C23, C33, frc.x);
    
    return cubic_hermite(CP0X, CP1X, CP2X, CP3X, frc.y);
}

float mitchell_netravali(float x) {
    float B = 1.0 / 3.0;
    float C = 1.0 / 3.0;

    float ax = abs(x);
    if (ax < 1) {
        return ((12 - 9 * B - 6 * C) * ax * ax * ax + (-18 + 12 * B + 6 * C) * ax * ax + (6 - 2 * B)) / 6;
    } else if ((ax >= 1) && (ax < 2)) {
        return ((-B - 6 * C) * ax * ax * ax + (6 * B + 30 * C) * ax * ax + (-12 * B - 48 * C) * ax + (8 * B + 24 * C)) / 6;
    } else {
        return 0;
    }
}

float3 fetch_center_filtered(int2 pix) {
    float4 res = 0.0.xxxx;

    int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            int2 src = pix + int2(x, y);
            float4 col = float4(decode(input_tex[src].rgb), 1);
            float dist = length(jitter * float2(1, -1) - float2(x, y));
            float wt = mitchell_netravali(dist);
            col *= wt;
            res += col;
        }
    }

    return res.rgb / res.a;
}


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    
    float3 center = decode(input_tex[px].rgb);
    center = rgb_to_ycbcr(center);

    const float4 reproj = reprojection_tex[px];
    float2 history_uv = uv + reproj.xy;

#if 1
    float history_g = fetch_history_catmull_rom(history_uv).y;
    float3 history = fetch_history(history_uv);
    if (history.y > 1e-5) {
        history *= history_g / history.y;
    }
#else
    float3 history = fetch_history_catmull_rom(history_uv);
#endif

    history = rgb_to_ycbcr(history);
    
	float3 vsum = 0.0.xxx;
	float3 vsum2 = 0.0.xxx;
	float wsum = 0;
    
	const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float3 neigh = decode(input_tex[px + int2(x, y)].rgb);
            neigh = rgb_to_ycbcr(neigh);

			float w = exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));
			vsum += neigh * w;
			vsum2 += neigh * neigh * w;
			wsum += w;
        }
    }

	float3 ex = vsum / wsum;
	float3 ex2 = vsum2 / wsum;
	float3 dev = sqrt(max(0.0.xxx, ex2 - ex * ex));

    float local_contrast = dev.x / (ex.x + 1e-5);

    float2 history_pixel = history_uv * output_tex_size.xy;
    float texel_center_dist = dot(1.0.xx, abs(0.5 - frac(history_pixel)));

    float box_size = 1.0;
    box_size *= lerp(0.5, 1.0, smoothstep(-0.1, 0.3, local_contrast));
    box_size *= lerp(0.5, 1.0, clamp(1.0 - texel_center_dist, 0.0, 1.0));

    //center = rgb_to_ycbcr(fetch_center_filtered(px));

    const float n_deviations = 1.5 * lerp(1.0, 0.5, reproj.w);
	float3 nmin = lerp(center, ex, box_size * box_size) - dev * box_size * n_deviations;
	float3 nmax = lerp(center, ex, box_size * box_size) + dev * box_size * n_deviations;

    float blend_factor = 1.0;
    
	#if 1
        // TODO: make better use of the quad reprojection validity
        uint quad_reproj_valid_packed = uint(reproj.z * 15.0 + 0.5);
        float4 quad_reproj_valid = (quad_reproj_valid_packed & uint4(1, 2, 4, 8)) != 0;
        blend_factor = lerp(1.0, 1.0 / 12.0, dot(quad_reproj_valid, 0.25));

        float3 clamped_history = clamp(history, nmin, nmax);
		//float3 clamped_history = history;//clamp(history, nmin, nmax);

        // "Anti-flicker"
        float clamp_dist = (min(abs(history.x - nmin.x), abs(history.x - nmax.x))) / max(max(history.x, ex.x), 1e-5);
        blend_factor *= lerp(0.2, 1.0, smoothstep(0.0, 2.0, clamp_dist));

		float3 result = lerp(clamped_history, center, blend_factor);
        result = ycbcr_to_rgb(result);

		result = encode(result);
	#else
        center = ycbcr_to_rgb(center);
		float3 result = encode(center);
	#endif

#if 0
    if (all(0 == px)) {
        result.x = int(history_tex[uint2(0, 0)].x + 1) % 255;
    }

    if (px.y > 0 && px.y < 40) {
        result = int(history_tex[uint2(0, 0)].x) == px.x / 6;
    }
#endif

    //result = float3(reproj.xy, 0);
    //uint quad_reproj_valid_packed = uint(reproj.z * 15.0 + 0.5);
    //float4 quad_reproj_valid = (quad_reproj_valid_packed & uint4(1, 2, 4, 8)) != 0;
    //result = quad_reproj_valid.rgb;

    output_tex[px] = float4(result, 1);
}
