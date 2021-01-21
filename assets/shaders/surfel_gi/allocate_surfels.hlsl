#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked
#include "../inc/color.hlsl"

Texture2D<float4> gbuffer_tex;
Texture2D<float> depth_tex;
RWByteAddressBuffer surfel_meta_buf;
RWBuffer<uint> surfel_hash_buf;
RWStructuredBuffer<VertexPacked> surfel_spatial_buf;

[numthreads(8, 8, 1)]
void main(in uint2 pix : SV_DispatchThreadID) {
    float4 output_tex_size = float4(1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0);
    float2 uv = get_uv(pix, output_tex_size);

    float4 gbuffer_packed = gbuffer_tex[pix];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        return;
    }

    float z_over_w = depth_tex[pix];
    float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, mul(frame_constants.view_constants.sample_to_view, pt_cs));
    pt_ws /= pt_ws.w;

    uint pt_hash = hash3(asuint(int3(floor(pt_ws.xyz * 3.0))));
    //total_radiance += uint_id_to_color(pt_hash);
}
