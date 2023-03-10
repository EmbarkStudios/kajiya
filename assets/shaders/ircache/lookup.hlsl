#ifndef IRCACHE_LOOKUP_HLSL
#define IRCACHE_LOOKUP_HLSL

#include "../inc/quasi_random.hlsl"
#include "../inc/sh.hlsl"

#include "ircache_grid.hlsl"
#include "ircache_sampler_common.inc.hlsl"

#define IRCACHE_LOOKUP_MAX 1

struct IrcacheLookup {
    uint entry_idx[IRCACHE_LOOKUP_MAX];
    float weight[IRCACHE_LOOKUP_MAX];
    uint count;
};

IrcacheLookup ircache_lookup(float3 pt_ws, float3 normal_ws, float3 jitter) {
    IrcacheLookup result;
    result.count = 0;

    const IrcacheCoord rcoord = ws_pos_to_ircache_coord(pt_ws, normal_ws, jitter);
    const uint cell_idx = rcoord.cell_idx();

    const uint2 cell_meta = ircache_grid_meta_buf.Load2(sizeof(uint2) * cell_idx);

    if (cell_meta.y & IRCACHE_ENTRY_META_OCCUPIED) {
        const uint entry_idx = cell_meta.x;
        result.entry_idx[result.count] = entry_idx;
        result.weight[result.count] = 1;
        result.count = 1;
    }

    return result;
}

struct IrcacheLookupMaybeAllocate {
    IrcacheLookup lookup;
    Vertex proposal;
    bool just_allocated;
};

struct IrcacheLookupParams {
    float3 query_from_ws;
    float3 pt_ws;
    float3 normal_ws;
    uint query_rank;
    bool stochastic_interpolation;

    static IrcacheLookupParams create(float3 query_from_ws, float3 pt_ws, float3 normal_ws) {
        IrcacheLookupParams res;
        res.query_from_ws = query_from_ws;
        res.pt_ws = pt_ws;
        res.normal_ws = normal_ws;
        res.query_rank = 0;
        res.stochastic_interpolation = false;
        return res;
    }

    IrcacheLookupParams with_query_rank(uint v) {
        IrcacheLookupParams res = this;
        res.query_rank = v;
        return res;
    }

    IrcacheLookupParams with_stochastic_interpolation(bool v) {
        IrcacheLookupParams res = this;
        res.stochastic_interpolation = v;
        return res;
    }

    float3 lookup(inout uint rng);
    IrcacheLookupMaybeAllocate lookup_maybe_allocate(inout uint rng);
};

IrcacheLookupMaybeAllocate IrcacheLookupParams::lookup_maybe_allocate(inout uint rng) {
    bool allocated_by_us = false;
    bool just_allocated = false;

    const float3 jitter = select(stochastic_interpolation
        , (float3(
            uint_to_u01_float(hash1_mut(rng)),
            uint_to_u01_float(hash1_mut(rng)),
            uint_to_u01_float(hash1_mut(rng))
        ) - 0.5)
        , 0.0.xxx);

#ifndef IRCACHE_LOOKUP_DONT_KEEP_ALIVE
    if (!IRCACHE_FREEZE) {
        const float3 eye_pos = get_eye_position();

        const IrcacheCoord rcoord = ws_pos_to_ircache_coord(pt_ws, normal_ws, jitter);

        const int3 scroll_offset = frame_constants.ircache_cascades[rcoord.cascade].voxels_scrolled_this_frame.xyz;
        const int3 was_just_scrolled_in =
            select(scroll_offset > 0
            , (int3(rcoord.coord) + scroll_offset >= IRCACHE_CASCADE_SIZE)
            , (int3(rcoord.coord) < -scroll_offset));

        // When a voxel is just scrolled in to a cascade, allocating it via indirect rays
        // has a good chance of creating leaks. Delay the allocation for one frame
        // unless we have a suitable one from a primary ray.
        const bool skip_allocation =
            query_rank >= IRCACHE_ENTRY_RANK_COUNT
            || (any(was_just_scrolled_in) && query_rank > 0);

        const uint cell_idx = rcoord.cell_idx();
        const uint2 cell_meta = ircache_grid_meta_buf.Load2(sizeof(uint2) * cell_idx);
        const uint entry_flags = cell_meta.y;

        just_allocated = (entry_flags & IRCACHE_ENTRY_META_JUST_ALLOCATED) != 0;

        if (!skip_allocation) {
            if ((entry_flags & IRCACHE_ENTRY_META_OCCUPIED) == 0) {
                // Allocate

                uint prev = 0;
                ircache_grid_meta_buf.InterlockedOr(
                    sizeof(uint2) * cell_idx + sizeof(uint),
                    IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_JUST_ALLOCATED,
                    prev);

                if ((prev & IRCACHE_ENTRY_META_OCCUPIED) == 0) {
                    // We've allocated it!
                    just_allocated = true;
                    allocated_by_us = true;

                    uint alloc_idx;
                    ircache_meta_buf.InterlockedAdd(IRCACHE_META_ALLOC_COUNT, 1, alloc_idx);

                    uint entry_idx = ircache_pool_buf[alloc_idx];
                    ircache_meta_buf.InterlockedMax(IRCACHE_META_ENTRY_COUNT, entry_idx + 1);

                    // Clear dead state, mark used.

                    ircache_life_buf.Store(entry_idx * 4, ircache_entry_life_for_rank(query_rank));
                    ircache_entry_cell_buf[entry_idx] = cell_idx;
                    ircache_grid_meta_buf.Store(sizeof(uint2) * cell_idx + 0, entry_idx);
                }
            }
        }
    }
#endif

    IrcacheLookup lookup = ircache_lookup(pt_ws, normal_ws, jitter);

    const uint cascade = ws_pos_to_ircache_coord(pt_ws, normal_ws, jitter).cascade;
    const float cell_diameter = ircache_grid_cell_diameter_in_cascade(cascade);

    float3 to_eye = normalize(get_eye_position() - pt_ws.xyz);
    float3 offset_towards_query = query_from_ws - pt_ws.xyz;
    const float MAX_OFFSET = cell_diameter;   // world units
    const float MAX_OFFSET_AS_FRAC = 0.5;   // fraction of the distance from query point
    offset_towards_query *= MAX_OFFSET / max(MAX_OFFSET / MAX_OFFSET_AS_FRAC, length(offset_towards_query));

    Vertex new_entry;
    #if IRCACHE_USE_SPHERICAL_HARMONICS
        // probes
        new_entry.position = pt_ws.xyz + offset_towards_query;
    #else
        // surface points
        new_entry.position = pt_ws.xyz;
    #endif
    new_entry.normal = normal_ws;
    //new_entry.normal = to_eye;

    if (allocated_by_us) {
        [unroll]
        for (uint i = 0; i < IRCACHE_LOOKUP_MAX; ++i) if (i < lookup.count) {
            const uint entry_idx = lookup.entry_idx[i];
            ircache_reposition_proposal_buf[entry_idx] = pack_vertex(new_entry);
        }
    }

    IrcacheLookupMaybeAllocate res;
    res.lookup = lookup;
    res.just_allocated = just_allocated;
    res.proposal = new_entry;
    return res;
}

float eval_sh_simplified(float4 sh, float3 normal) {
    float4 lobe_sh = float4(0.8862, 1.0233 * normal);
    return dot(sh, lobe_sh);
}

float eval_sh_geometrics(float4 sh, float3 normal)
{
	// http://www.geomerics.com/wp-content/uploads/2015/08/CEDEC_Geomerics_ReconstructingDiffuseLighting1.pdf

	float R0 = sh.x;

	float3 R1 = 0.5f * float3(sh.y, sh.z, sh.w);
	float lenR1 = length(R1);

	float q = 0.5f * (1.0f + dot(R1 / lenR1, normal));

	float p = 1.0f + 2.0f * lenR1 / R0;
	float a = (1.0f - lenR1 / R0) / (1.0f + lenR1 / R0);

	return R0 * (a + (1.0f - a) * (p + 1.0f) * pow(q, p));
}

float eval_sh_nope(float4 sh, float3 normal) {
    return sh.x / (0.282095 * 4);
}

#if IRCACHE_USE_SPHERICAL_HARMONICS
    #if 0
        #define eval_sh eval_sh_simplified
    #else
        #define eval_sh eval_sh_geometrics
    #endif
#else
    #define eval_sh eval_sh_nope
#endif


float3 IrcacheLookupParams::lookup(inout uint rng) {
    IrcacheLookupMaybeAllocate lookup = lookup_maybe_allocate(rng);

    if (lookup.just_allocated) {
        return 0.0.xxx;
    }

    float3 irradiance_sum = 0.0.xxx;

#ifdef IRCACHE_LOOKUP_KEEP_ALIVE_PROB
    const bool should_propose_position = uint_to_u01_float(hash1_mut(rng)) < IRCACHE_LOOKUP_KEEP_ALIVE_PROB;
#else
    const bool should_propose_position = true;
#endif

    [unroll]
    for (uint i = 0; i < IRCACHE_LOOKUP_MAX; ++i) if (i < lookup.lookup.count) {
        const uint entry_idx = lookup.lookup.entry_idx[i];
        
        float3 irradiance = 0;

#ifdef IRCACHE_LOOKUP_PRECISE
        {
            float weight_sum = 0;

            // TODO: counter distortion
            for (uint octa_idx = 0; octa_idx < IRCACHE_OCTA_DIMS2; ++octa_idx) {
                const float2 octa_coord = (float2(octa_idx % IRCACHE_OCTA_DIMS, octa_idx / IRCACHE_OCTA_DIMS) + 0.5) / IRCACHE_OCTA_DIMS;

                const Reservoir1spp r = Reservoir1spp::from_raw(asuint(ircache_aux_buf[entry_idx * IRCACHE_AUX_STRIDE + octa_idx].xy));
                const float3 dir = SampleParams::from_raw(r.payload).direction();

                const float wt = dot(dir, normal_ws);
                if (wt > 0.0) {
                    const float4 contrib = ircache_aux_buf[entry_idx * IRCACHE_AUX_STRIDE + IRCACHE_OCTA_DIMS2 + octa_idx];
                    irradiance += contrib.rgb * wt * contrib.w;
                    weight_sum += wt;
                }
            }

            irradiance /= max(1.0, weight_sum);
        }
#else
        for (uint basis_i = 0; basis_i < 3; ++basis_i) {
            irradiance[basis_i] += eval_sh(ircache_irradiance_buf[entry_idx * IRCACHE_IRRADIANCE_STRIDE + basis_i], normal_ws);
        }
#endif

        irradiance = max(0.0.xxx, irradiance);
        irradiance_sum += irradiance * lookup.lookup.weight[i];

        if (!IRCACHE_FREEZE && should_propose_position) {
            #ifndef IRCACHE_LOOKUP_DONT_KEEP_ALIVE
                const uint prev_life = ircache_life_buf.Load(entry_idx * 4);

                if (prev_life < IRCACHE_ENTRY_LIFE_RECYCLE) {
                    const uint new_life = ircache_entry_life_for_rank(query_rank);
                    if (new_life < prev_life) {
                        ircache_life_buf.InterlockedMin(entry_idx * 4, new_life);
                        //ircache_life_buf.Store(entry_idx * 4, new_life);
                    }

                    const uint prev_rank = ircache_entry_life_to_rank(prev_life);
                    if (query_rank <= prev_rank) {
                        uint prev_vote_count;
                        InterlockedAdd(ircache_reposition_proposal_count_buf[entry_idx], 1, prev_vote_count);

                        const float dart = uint_to_u01_float(hash1_mut(rng));
                        const float prob = 1.0 / (prev_vote_count + 1.0);

                        if (!IRCACHE_USE_UNIFORM_VOTING || dart <= prob) {
                            ircache_reposition_proposal_buf[entry_idx] = pack_vertex(lookup.proposal);
                        }
                    }
                }
            #endif
        }

        //irradiance_sum = (entry_idx % 64) / 63.0;
    }

    return irradiance_sum;
}

#endif // IRCACHE_LOOKUP_HLSL
