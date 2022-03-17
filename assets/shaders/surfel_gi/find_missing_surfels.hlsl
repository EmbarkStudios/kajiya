#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/mesh.hlsl" // for VertexPacked
#include "../inc/gbuffer.hlsl"
#include "../inc/color.hlsl"
#include "../inc/sh.hlsl"

#define VISUALIZE_ENTRIES 0
#define VISUALIZE_CASCADES 0
#define VISUALIZE_SURFEL_AGE 0
#define VISUALIZE_CELLS 0
#define USE_GEOMETRIC_NORMALS 1
#define FREEZE_SURFEL_SET 0
#define USE_DEBUG_OUT 1

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> geometric_normals_tex;
[[vk::binding(3)]] RWByteAddressBuffer surf_rcache_meta_buf;
[[vk::binding(4)]] RWByteAddressBuffer surf_rcache_grid_meta_buf;
[[vk::binding(5)]] StructuredBuffer<VertexPacked> surf_rcache_spatial_buf;
[[vk::binding(6)]] StructuredBuffer<float4> surf_rcache_irradiance_buf;
[[vk::binding(7)]] RWTexture2D<float4> debug_out_tex;
[[vk::binding(8)]] RWStructuredBuffer<uint> surf_rcache_pool_buf;
[[vk::binding(9)]] RWStructuredBuffer<uint> surf_rcache_life_buf;
[[vk::binding(10)]] RWStructuredBuffer<VertexPacked> surf_rcache_reposition_proposal_buf;

[[vk::binding(11)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

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

float eval_sh_simplified(float4 sh, float3 normal) {
    //return (0.5 + dot(sh.xyz, normal)) * sh.w * 2;
    //return sh.w;

    float4 lobe_sh = float4(0.8862, 1.0233 * normal);
    return dot(sh * float4(1, sh.xxx), lobe_sh) * M_PI;
    //return sh.x * 4;
}

float eval_sh_geometrics(float4 sh, float3 normal)
{
	// http://www.geomerics.com/wp-content/uploads/2015/08/CEDEC_Geomerics_ReconstructingDiffuseLighting1.pdf

	float R0 = sh.x;

	float3 R1 = 0.5f * float3(sh.y, sh.z, sh.w) * sh.x;
	float lenR1 = length(R1);

	float q = 0.5f * (1.0f + dot(R1 / lenR1, normal));

	float p = 1.0f + 2.0f * lenR1 / R0;
	float a = (1.0f - lenR1 / R0) / (1.0f + lenR1 / R0);

	return R0 * (a + (1.0f - a) * (p + 1.0f) * pow(q, p)) * M_PI;
}

uint pack_score_and_px_within_group(float score, uint2 px_within_group) {
    return (asuint(score) & (0xffffffffu - 63)) | (px_within_group.y * 8 + px_within_group.x);
}

#if 1
    #define eval_surfel_sh eval_sh_simplified
#else
    #define eval_surfel_sh eval_sh_geometrics
#endif

[numthreads(8, 8, 1)]
void main(
    uint2 px: SV_DispatchThreadID,
    uint idx_within_group: SV_GroupIndex,
    uint2 group_id: SV_GroupID,
    uint2 px_within_group: SV_GroupThreadID
) {
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

    uint seed = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);
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
    float3 debug_color = lookup_surfel_gi(pt_ws.xyz, gbuffer.normal);

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

                surf_rcache_grid_meta_buf.Store(sizeof(uint4) * cell_idx + 0, entry_idx);
            } else {
                // We did not allocate it, so read the entry index from whoever did.
                
                entry_idx = surf_rcache_grid_meta_buf.Load(sizeof(uint4) * cell_idx + 0);
            }
        }

        // TODO: reservoir-based selection, factor in vertex ordinals
        {
            Vertex new_surfel;
            new_surfel.position = pt_ws.xyz;
            new_surfel.normal = shading_normal;
            surf_rcache_reposition_proposal_buf[entry_idx] = pack_vertex(new_surfel);
        }
        
        float4 surfel_irradiance_packed = surf_rcache_irradiance_buf[entry_idx];
        surfel_color = surfel_irradiance_packed.xyz;

        #if VISUALIZE_ENTRIES
            debug_color = surfel_color;
        #endif
   
        #if VISUALIZE_SURFEL_AGE
            //debug_color = cost_color_map(1.4 - min(1.0, pow(surfel_irradiance_packed.w / 128.0, 8.0)));
        #endif
    }

    // Despawn surfels
    /*if (!FREEZE_SURFEL_SET) {
        uint px_max_score_loc_packed = 0u;

        const float cell_fullness = smoothstep(
            MAX_SURFELS_PER_CELL_FOR_KEEP_ALIVE * 0.75,
            MAX_SURFELS_PER_CELL_FOR_KEEP_ALIVE * 1.0,
            float(cell_surfel_count));

        // Be more fussy about high coverage if the surfel count is getting high
        const float despawn_weight_threshold = lerp(3.5, 3.0, cell_fullness);
        const float second_highest_weight_threshold = lerp(0.9, 0.8, cell_fullness);

        // Despawn if the coverage is high" and the second highest scoring
        // surfel has a high weight (indicating surfel overlap),
        if (scoring_total_weight > despawn_weight_threshold && second_highest_weight > second_highest_weight_threshold) {
            px_max_score_loc_packed = pack_score_and_px_within_group(px_score, px_within_group);
            InterlockedMax(gs_px_max_score_loc_packed, px_max_score_loc_packed);
        }

        GroupMemoryBarrierWithGroupSync();

        uint surfel_to_despawn = 0xffffffffu;
        float surfel_to_despawn_weight = 0;

        if (gs_px_max_score_loc_packed == px_max_score_loc_packed && px_max_score_loc_packed != 0) {
            for (uint entry_idx_loc = entry_idx_loc_range.x; entry_idx_loc < entry_idx_loc_range.y; ++entry_idx_loc) {
                const uint entry_idx = surfel_index_buf.Load(sizeof(uint) * entry_idx_loc);

                Vertex surfel = unpack_vertex(surf_rcache_spatial_buf[entry_idx]);
                const float surfel_radius = surfel_radius_for_pos(surfel.position);

            #if USE_GEOMETRIC_NORMALS
                float3 shading_normal = geometric_normal_ws;
            #else
                float3 shading_normal = gbuffer.normal;
            #endif

                const float3 pos_offset = pt_ws.xyz - surfel.position.xyz;
                const float directional_weight = max(0.0, dot(surfel.normal, shading_normal));
                const float dist = length(pos_offset);
                const float mahalanobis_dist = length(pos_offset) * (1 + abs(dot(pos_offset, surfel.normal)) * SURFEL_NORMAL_DIRECTION_SQUISH);

                float weight = smoothstep(
                    surfel_radius * SURFEl_RADIUS_OVERSCALE,
                    0.0,
                    mahalanobis_dist) * directional_weight;

                if (weight > surfel_to_despawn_weight) {
                    surfel_to_despawn_weight = weight;
                    surfel_to_despawn = entry_idx;
                }
            }
        }

        if (surfel_to_despawn != 0xffffffffu) {
            surf_rcache_life_buf[surfel_to_despawn] = SURFEL_LIFE_RECYCLE;
        }
    }*/

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
