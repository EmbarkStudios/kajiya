#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked
#include "../inc/color.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] RWByteAddressBuffer surfel_meta_buf;
[[vk::binding(3)]] RWByteAddressBuffer surfel_hash_buf;
[[vk::binding(4)]] RWStructuredBuffer<VertexPacked> surfel_spatial_buf;
[[vk::binding(5)]] RWTexture2D<float4> debug_out_tex;

static const uint MAX_SURFEL_CELLS = 1024 * 1024;

uint surfel_pos_to_hash(float3 pos) {
    return hash3(asuint(int3(floor(pos * 3.0))));
}

struct HashEntry {
    bool found;
    bool vacant;

    uint hash;
    uint addr;
    uint payload;

    bool acquire() {
        uint prev_value;
        surfel_hash_buf.InterlockedCompareExchange(addr, 0, hash, prev_value);
        return prev_value == 0;
    }
    
    void insert(uint payload) {
        surfel_hash_buf.Store(addr+4, payload);
    }
};

HashEntry surfel_hash_lookup(float3 pos) {
    const uint hash = surfel_pos_to_hash(pos);
    uint addr = (hash % MAX_SURFEL_CELLS) * 8;

    static const uint MAX_PROBE = 4;

    for (uint i = 0; i < MAX_PROBE; ++i, addr += 8) {
        const uint entry_hash = surfel_hash_buf.Load(addr);
        if (0 == entry_hash) {
            HashEntry res;
            res.found = false;
            res.vacant = true;
            res.hash = hash;
            res.addr = addr;
            return res;
        }

        if (entry_hash == hash) {
            HashEntry res;
            res.found = true;
            res.payload = surfel_hash_buf.Load(addr+4);
            res.addr = addr;
            return res;
        }
    }

    HashEntry res;
    res.found = false;
    res.vacant = false;
    res.addr = addr;
    return res;
}

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

    uint pt_hash = surfel_pos_to_hash(pt_ws.xyz);
    HashEntry entry = surfel_hash_lookup(pt_ws.xyz);

    float3 out_color = 0.5.xxx;

    if (entry.found) {
        out_color = uint_id_to_color(entry.payload) * float(entry.payload) / 100000.0;
    } else {
        if (entry.vacant) {
            if (uint_to_u01_float(hash1_mut(seed)) < 0.001) {
                if (entry.acquire()) {
                    uint surfel_idx = 0;
                    surfel_meta_buf.InterlockedAdd(0, 1, surfel_idx);
                    entry.insert(surfel_idx);
                }
            }
        } else {
            out_color = float3(10, 0, 10);
        }
    }

    debug_out_tex[px] = float4(out_color, 0);
}
