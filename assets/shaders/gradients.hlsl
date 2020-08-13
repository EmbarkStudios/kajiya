#include "inc/color.hlsl"
#include "inc/uv.hlsl"

RWTexture2D<float4> output_tex;

/*cbuffer globals {
    float4 output_tex_size;
    float time;
};*/

[numthreads(8, 8, 1)]
void main(in uint2 pix : SV_DispatchThreadID) {
    // TODO
    //float4 output_tex_size = float4(512.0, 512.0, 1.0 / 512.0, 1.0 / 512.0);
    float4 output_tex_size = float4(1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0);
    float time = 0.0;

	float2 uv = frac(get_uv(pix, output_tex_size) + float2(0.0, time));
	float hue = frac(int(uv.y * 6) / 6.0 + 0.09);
	float4 col = float4(hsv_to_rgb(float3(hue, 1.0, 1)) * uv.x, 1);
    output_tex[pix] = col;
}
