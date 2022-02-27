#include "../inc/frame_constants.hlsl"
#include "../inc/samplers.hlsl"

[[vk::binding(0)]] Texture2D<float> depth_tex;
[[vk::binding(1)]] Texture2D<float3> color_tex;
[[vk::binding(2)]] Texture2D<float> coc_tex;
[[vk::binding(3)]] Texture2D<float> coc_tiles_tex;
[[vk::binding(4)]] RWTexture2D<float3> output_tex;
[[vk::binding(5)]] cbuffer _ {
    float4 output_tex_size;
};

static const float GOLDEN_ANGLE = 2.39996323; 
static const float MAX_BLUR_SIZE = 20.0; 
static const float RAD_SCALE = 0.4; // Smaller = nicer blur, larger = faster

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = float2(px + 0.5) * output_tex_size.zw;
    float3 color = color_tex[px];

	float center_depth = depth_tex[px];
	float center_size = abs(coc_tex[px]);
	float tot = 1.0;
	float radius = RAD_SCALE;

    //float max_blur_size = MAX_BLUR_SIZE;
    float max_blur_size = 0;
    int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            max_blur_size = max(max_blur_size, coc_tiles_tex[px / 8 + int2(x, y)]);
        }
    }
    
#if 1
	for (float ang = 0.0; radius < max_blur_size; ang += GOLDEN_ANGLE) {
		float2 tc = uv + float2(cos(ang), sin(ang)) * output_tex_size.zw * radius;
		float3 sampleColor = color_tex.SampleLevel(sampler_lnc, tc, 0);
		float sampleDepth = depth_tex.SampleLevel(sampler_lnc, tc, 0);
		float sampleSize = abs(coc_tex.SampleLevel(sampler_lnc, tc, 0));

		if (sampleDepth < center_depth) {
			sampleSize = clamp(sampleSize, 0.0, center_size*2.0);
        }

		float m = smoothstep(radius-0.5, radius+0.5, sampleSize);
		color += lerp(color/tot, sampleColor, m);
		tot += 1.0;
        radius += RAD_SCALE/radius;
	}
#else
    static const int sample_count = int(max_blur_size) * 6;

    float ang = 0.0;
	for (float i = 0; i < sample_count; ++i) {
        float r = (i + 0.5) / sample_count;
        r = sqrt(r);
        r *= max_blur_size;

		float2 tc = uv + float2(cos(ang), sin(ang)) * output_tex_size.zw * r;
		float3 sampleColor = color_tex.SampleLevel(sampler_lnc, tc, 0);
		float sampleDepth = depth_tex.SampleLevel(sampler_lnc, tc, 0);
		float sampleSize = abs(coc_tex.SampleLevel(sampler_lnc, tc, 0));

		if (sampleDepth < center_depth) {
			sampleSize = clamp(sampleSize, 0.0, center_size*2.0);
        }

		float m = smoothstep(r-0.5, r+0.5, sampleSize);
		color += lerp(color/tot, sampleColor, m);
		tot += 1.0;
        ang += GOLDEN_ANGLE;
	}
#endif

    color /= tot;
    //color = neutral_tonemap(color);
    //color = 1 - exp(-color);
    //color = pow(color, 1.05);

    output_tex[px] = color;
}
