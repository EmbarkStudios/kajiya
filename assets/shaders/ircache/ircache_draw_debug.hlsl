#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked
#include "../inc/gbuffer.hlsl"
#include "../inc/color.hlsl"
#include "../inc/sh.hlsl"

#define VISUALIZE_ENTRIES 1
#define VISUALIZE_CASCADES 0
#define USE_GEOMETRIC_NORMALS 1
#define USE_DEBUG_OUT 1

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> geometric_normals_tex;
[[vk::binding(3)]] RWByteAddressBuffer ircache_meta_buf;
[[vk::binding(4)]] RWByteAddressBuffer ircache_grid_meta_buf;
[[vk::binding(5)]] RWStructuredBuffer<uint> ircache_entry_cell_buf;
[[vk::binding(6)]] StructuredBuffer<VertexPacked> ircache_spatial_buf;
[[vk::binding(7)]] StructuredBuffer<float4> ircache_irradiance_buf;
[[vk::binding(8)]] RWTexture2D<float4> debug_out_tex;
[[vk::binding(9)]] RWStructuredBuffer<uint> ircache_pool_buf;
[[vk::binding(10)]] RWStructuredBuffer<uint> ircache_life_buf;
[[vk::binding(11)]] RWStructuredBuffer<VertexPacked> ircache_reposition_proposal_buf;
[[vk::binding(12)]] RWStructuredBuffer<uint> ircache_reposition_proposal_count_buf;

[[vk::binding(13)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

#define IRCACHE_LOOKUP_DONT_KEEP_ALIVE
#include "lookup.hlsl"

groupshared uint gs_px_min_score_loc_packed;
groupshared uint gs_px_max_score_loc_packed;

float3 tricolor_ramp(float3 a, float3 b, float3 c, float x) {
    x = saturate(x);

    #if 0
        float x2 = x * x;
        float x3 = x * x * x;
        float x4 = x2 * x2;

        float cw = 3*x2 - 2*x3;
        float aw = 1 - cw;
        float bw = 16*x4 - 32*x3 + 16*x2;
        float ws = aw + bw + cw;
        aw /= ws;
        bw /= ws;
        cw /= ws;
    #else
        const float lobe = pow(smoothstep(1, 0, 2 * abs(x - 0.5)), 2.5254);
        float aw = select(x < 0.5, 1 - lobe, 0);
        float bw = lobe;
        float cw = select(x > 0.5, 1 - lobe, 0);
    #endif

    return aw * a + bw * b + cw * c;
}

float3 cost_color_map(float x) {
    return tricolor_ramp(
        float3(0.05, 0.2, 1),
        float3(0.02, 1, 0.2),
        float3(1, 0.01, 0.1),
        x
    );
}

uint pack_score_and_px_within_group(float score, uint2 px_within_group) {
    return (asuint(score) & (0xffffffffu - 63)) | (px_within_group.y * 8 + px_within_group.x);
}

[numthreads(8, 8, 1)]
void main(
    uint2 px: SV_DispatchThreadID,
    uint idx_within_group: SV_GroupIndex,
    uint2 group_id: SV_GroupID,
    uint2 px_within_group: SV_GroupThreadID
) {
    //return;
    
    const float3 prev_eye_pos = get_prev_eye_position();

    if (USE_DEBUG_OUT && px.y < 50) {
        const uint entry_count = ircache_meta_buf.Load(IRCACHE_META_ENTRY_COUNT);
        const uint entry_alloc_count = ircache_meta_buf.Load(IRCACHE_META_ALLOC_COUNT);
        
        const float u = float(px.x + 0.5) * gbuffer_tex_size.z;

        if (px.y < 25) {
            if (entry_alloc_count > u * 256 * 1024) {
                debug_out_tex[px] = float4(0.05, 1, .2, 1);
                return;
            }
        } else {
            if (entry_count > u * 256 * 1024) {
                debug_out_tex[px] = float4(1, 0.1, 0.05, 1);
                return;
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    uint rng = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);
    const float2 uv = get_uv(px, gbuffer_tex_size);

    #if USE_DEBUG_OUT
        debug_out_tex[px] = 0.0.xxxx;
    #endif

    const float z_over_w = depth_tex[px];

    if (z_over_w == 0.0) {
        return;
    }

    // TODO: nuke
    const float4 gbuffer_packed = gbuffer_tex[px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();

    const float3 geometric_normal_ws = mul(
        frame_constants.view_constants.view_to_world,
        float4(geometric_normals_tex[px].xyz * 2 - 1, 0)
    ).xyz;

    const float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    const float4 pt_vs = mul(frame_constants.view_constants.sample_to_view, pt_cs);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, pt_vs);
    pt_ws /= pt_ws.w;

    const float pt_depth = -pt_vs.z / pt_vs.w;

   #if USE_GEOMETRIC_NORMALS
        const float3 shading_normal = geometric_normal_ws;
    #else
        const float3 shading_normal = gbuffer.normal;
    #endif

    float3 debug_color = IrcacheLookupParams::create(get_eye_position(), pt_ws.xyz, gbuffer.normal).lookup(rng);

    #if USE_DEBUG_OUT
        debug_out_tex[px] = float4(debug_color, 1);
    #endif
}
