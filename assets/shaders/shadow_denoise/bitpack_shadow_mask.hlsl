[[vk::binding(0)]] Texture2D<float> input_tex;
[[vk::binding(1)]] RWTexture2D<uint> output_tex;
[[vk::binding(2)]] cbuffer _ {
    float4 input_tex_size;
    uint2 bitpacked_shadow_mask_extent;
};

uint2 FFX_DNSR_Shadows_GetBufferDimensions() {
    return uint2(input_tex_size.xy);
}

bool FFX_DNSR_Shadows_HitsLight(uint2 px, uint2 gtid, uint2 gid) {
    return input_tex[px] > 0.5;
}

void FFX_DNSR_Shadows_WriteMask(uint linear_tile_index, uint value) {
    const uint2 tile = uint2(
        linear_tile_index % bitpacked_shadow_mask_extent.x,
        linear_tile_index / bitpacked_shadow_mask_extent.x
    );

    output_tex[tile] = value;
}

#include "ffx/ffx_denoiser_shadows_prepare.hlsl"

[numthreads(8, 4, 1)]
void main(uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID) {
    FFX_DNSR_Shadows_PrepareShadowMask(gtid, gid);
}
