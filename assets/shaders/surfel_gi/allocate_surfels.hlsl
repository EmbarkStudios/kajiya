#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked
#include "../inc/color.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] RWByteAddressBuffer surfel_meta_buf;
[[vk::binding(3)]] RWByteAddressBuffer surfel_hash_key_buf;
[[vk::binding(4)]] RWByteAddressBuffer surfel_hash_value_buf;
[[vk::binding(5)]] RWStructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(6)]] RWTexture2D<float4> debug_out_tex;

#include "surfel_grid_hash.hlsl"

[numthreads(8, 8, 1)]
void main(in uint2 px: SV_DispatchThreadID) {
    uint seed = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);

    float4 output_tex_size = float4(1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0);
    float2 uv = get_uv(px, output_tex_size);

    float4 gbuffer_packed = gbuffer_tex[px];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        debug_out_tex[px] = 0.0.xxxx;
        return;
    }

    float z_over_w = depth_tex[px];
    float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, mul(frame_constants.view_constants.sample_to_view, pt_cs));
    pt_ws /= pt_ws.w;

    SurfelGridHashEntry entry = surfel_hash_lookup(pt_ws.xyz);

    float3 out_color = 0.5.xxx;

    if (entry.found) {
        const uint value = surfel_hash_value_buf.Load(entry.idx * 4);
        out_color = uint_id_to_color(value) * float(value) / 100000.0;
    } else {
        if (entry.vacant) {
            if (uint_to_u01_float(hash1_mut(seed)) < 0.001) {
                if (entry.acquire()) {
                    uint cell_idx = 0;
                    surfel_meta_buf.InterlockedAdd(0, 1, cell_idx);

                    surfel_hash_value_buf.Store(entry.idx * 4, cell_idx);
                }
            }
        } else {
            // Too many conflicts; cannot insert a new entry.
            debug_out_tex[px] = float4(10, 0, 10, 1);
            return;
        }
    }

    debug_out_tex[px] = float4(out_color, 1);
}
