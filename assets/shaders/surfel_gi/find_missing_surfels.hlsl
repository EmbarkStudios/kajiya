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
#define VISUALIZE_SURFEL_AGE 0
#define VISUALIZE_CELLS 0
#define USE_GEOMETRIC_NORMALS 1
#define USE_DEBUG_OUT 1

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> geometric_normals_tex;
[[vk::binding(3)]] RWByteAddressBuffer surf_rcache_meta_buf;
[[vk::binding(4)]] RWByteAddressBuffer surf_rcache_grid_meta_buf;
[[vk::binding(5)]] RWStructuredBuffer<uint> surf_rcache_entry_cell_buf;
[[vk::binding(6)]] StructuredBuffer<VertexPacked> surf_rcache_spatial_buf;
[[vk::binding(7)]] StructuredBuffer<float4> surf_rcache_irradiance_buf;
[[vk::binding(8)]] RWTexture2D<float4> debug_out_tex;
[[vk::binding(9)]] RWStructuredBuffer<uint> surf_rcache_pool_buf;
[[vk::binding(10)]] RWStructuredBuffer<uint> surf_rcache_life_buf;
[[vk::binding(11)]] RWStructuredBuffer<VertexPacked> surf_rcache_reposition_proposal_buf;
[[vk::binding(12)]] RWStructuredBuffer<uint> surf_rcache_reposition_proposal_count_buf;

[[vk::binding(13)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

// TODO: figure out which one actually works better
//#define SURFEL_LOOKUP_DONT_KEEP_ALIVE
#define SURFEL_LOOKUP_KEEP_ALIVE_PROB 0.05

#include "lookup.hlsl"
#include "surfel_binning_shared.hlsl"

groupshared uint gs_px_min_score_loc_packed;
groupshared uint gs_px_max_score_loc_packed;

float inverse_lerp(float minv, float maxv, float v) {
    return (v - minv) / (maxv - minv);
}

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
        float aw = x < 0.5 ? 1 - lobe : 0;
        float bw = lobe;
        float cw = x > 0.5 ? 1 - lobe : 0;
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
        const uint surfel_count = surf_rcache_meta_buf.Load(SURFEL_META_ENTRY_COUNT);
        const uint surfel_alloc_count = surf_rcache_meta_buf.Load(SURFEL_META_ALLOC_COUNT);
        
        const float u = float(px.x + 0.5) * gbuffer_tex_size.z;

        if (px.y < 25) {
            if (surfel_alloc_count > u * 256 * 1024) {
                debug_out_tex[px] = float4(0.05, 1, .2, 1);
                return;
            }
        } else {
            if (surfel_count > u * 256 * 1024) {
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

    const uint cell_idx = surfel_grid_coord_to_hash(surfel_pos_to_grid_coord(pt_ws.xyz, prev_eye_pos));

    const uint4 pt_c4 = surfel_grid_coord_to_c4(surfel_pos_to_grid_coord(pt_ws.xyz, prev_eye_pos));
    const uint pt_c4_hash = surfel_grid_c4_to_hash(pt_c4);

   #if USE_GEOMETRIC_NORMALS
        const float3 shading_normal = geometric_normal_ws;
    #else
        const float3 shading_normal = gbuffer.normal;
    #endif

    float3 surfel_color = 0.0.xxx;
    float3 debug_color = lookup_surfel_gi(get_eye_position(), pt_ws.xyz, gbuffer.normal, 0, rng);

#if 0
    {
        const uint4 cell_meta = surf_rcache_grid_meta_buf.Load4(sizeof(uint4) * cell_idx);
        uint entry_idx = cell_meta.x;
        const uint entry_flags = cell_meta.y;

        if ((entry_flags & SURF_RCACHE_ENTRY_META_OCCUPIED) == 0) {
            // Allocate

            uint prev = 0;
            surf_rcache_grid_meta_buf.InterlockedOr(sizeof(uint4) * cell_idx + sizeof(uint), SURF_RCACHE_ENTRY_META_OCCUPIED, prev);

            if ((prev & SURF_RCACHE_ENTRY_META_OCCUPIED) == 0) {
                // We've allocated it!

                uint alloc_idx;
                surf_rcache_meta_buf.InterlockedAdd(SURFEL_META_ALLOC_COUNT, 1, alloc_idx);

                entry_idx = surf_rcache_pool_buf[alloc_idx];
                surf_rcache_meta_buf.InterlockedMax(SURFEL_META_ENTRY_COUNT, entry_idx + 1);

                // Clear dead state, mark used.
                surf_rcache_life_buf[entry_idx] = 0;
                surf_rcache_entry_cell_buf[entry_idx] = cell_idx;

                surf_rcache_grid_meta_buf.Store(sizeof(uint4) * cell_idx + 0, entry_idx);
            } else {
                // We did not allocate it, so read the entry index from whoever did.
                
                entry_idx = surf_rcache_grid_meta_buf.Load(sizeof(uint4) * cell_idx + 0);
            }
        }

        // TODO: reservoir-based selection, factor in vertex ranks
        if (!true) {
            SurfRcacheLookup lookup = surf_rcache_lookup(pt_ws.xyz);
            Vertex new_surfel;
            new_surfel.position = pt_ws.xyz;
            new_surfel.normal = shading_normal;

            #if 1
            [unroll]
            for (uint i = 0; i < SURF_CACHE_LOOKUP_MAX; ++i) if (i < lookup.count) {
                const uint entry_idx = lookup.entry_idx[i];

                // HACK; TODO: only accept trilinear footprint proposals if no direct proposals found
                // or direct proposals are from a lower rank
                const float prob = pow(lookup.weight[i], 3);
                if (uint_to_u01_float(hash1_mut(rng)) > prob) {
                    continue;
                }

            #else
            if (true) {
            #endif
                surf_rcache_reposition_proposal_buf[entry_idx] = pack_vertex(new_surfel);

                // Mark used
                if (surf_rcache_life_buf[entry_idx] < SURFEL_LIFE_RECYCLE) {
                    surf_rcache_life_buf[entry_idx] = 0;
                }
            }
        }
        
        float4 surfel_irradiance_packed = surf_rcache_irradiance_buf[entry_idx * SURF_RCACHE_IRRADIANCE_STRIDE];
        surfel_color = 1;//surfel_irradiance_packed.xyz;

        #if VISUALIZE_ENTRIES
            debug_color = surfel_color;
        #endif
   
        #if VISUALIZE_SURFEL_AGE
            //debug_color = cost_color_map(1.4 - min(1.0, pow(surfel_irradiance_packed.w / 128.0, 8.0)));
        #endif
    }
#endif

    #if VISUALIZE_CASCADES
        Vertex surfel;
        surfel.position = pt_ws.xyz;
        SurfelGridMinMax box = get_surfel_grid_box_min_max(surfel);

        debug_color = cost_color_map((
            surfel_grid_coord_to_cascade(surfel_pos_to_grid_coord(pt_ws.xyz, prev_eye_pos))
            + 1) / 8.0
        );

        debug_color = cost_color_map(
            (box.c4_min[0].w + 1) / 8.0
        );

        debug_color = cost_color_map(
            (box.c4_min[0].x % 32 + 1) / 32.0
        );

        if (box.cascade_count > 1) {
            debug_color = 1;
        }
    #endif

    if (VISUALIZE_CELLS) {
        const uint h = hash4(pt_c4);
        debug_color = uint_id_to_color(h);
    }
    //debug_color = uint_id_to_color(cell_idx) * 0.3;
    //debug_color = saturate(1.0 - length(pt_ws.xyz));

    #if USE_DEBUG_OUT
        debug_out_tex[px] = float4(debug_color, 1);
    #endif
}
