#include "inc/color.hlsl"

[[vk::binding(0)]] Texture2D<float4> ui_tex;
[[vk::binding(1)]] RWTexture2D<float4> main_tex;


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float4 main = main_tex[px];
    float4 gui = ui_tex[px];
    gui.rgb = srgb_to_linear(gui.rgb);

    float4 result = main;
    result.rgb = result.rgb * (1.0 - gui.a) + gui.rgb;

    main_tex[px] = result;
}
