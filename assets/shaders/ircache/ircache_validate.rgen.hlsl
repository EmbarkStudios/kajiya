#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/reservoir.hlsl"
#include "../inc/bindless_textures.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/mesh.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../wrc/bindings.hlsl"
#include "../inc/color.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] StructuredBuffer<VertexPacked> ircache_spatial_buf;
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] RWByteAddressBuffer ircache_grid_meta_buf;
[[vk::binding(3)]] RWByteAddressBuffer ircache_life_buf;
[[vk::binding(4)]] RWStructuredBuffer<VertexPacked> ircache_reposition_proposal_buf;
[[vk::binding(5)]] RWStructuredBuffer<uint> ircache_reposition_proposal_count_buf;
DEFINE_WRC_BINDINGS(6)
[[vk::binding(7)]] RWByteAddressBuffer ircache_meta_buf;
[[vk::binding(8)]] RWStructuredBuffer<float4> ircache_aux_buf;
[[vk::binding(9)]] RWStructuredBuffer<uint> ircache_pool_buf;
[[vk::binding(10)]] StructuredBuffer<uint> ircache_entry_indirection_buf;
[[vk::binding(11)]] RWStructuredBuffer<uint> ircache_entry_cell_buf;

#include "../inc/sun.hlsl"
#include "../wrc/lookup.hlsl"

// Sample straight from the `ircache_aux_buf` instead of the SH.
#define IRCACHE_LOOKUP_PRECISE
#include "lookup.hlsl"

#include "ircache_sampler_common.inc.hlsl"
#include "ircache_trace_common.inc.hlsl"

[shader("raygeneration")]
void main() {
    if (IRCACHE_FREEZE) {
        return;
    }

    const uint dispatch_idx = DispatchRaysIndex().x;

    // AMD ray-tracing bug workaround; indirect RT seems to be tracing with the same
    // ray count for multiple dispatches (???)
    // Search for c804a814-fdc8-4843-b2c8-9d0674c10a6f for other occurences.
    #if 1
        const uint alloc_count = ircache_meta_buf.Load(IRCACHE_META_TRACING_ALLOC_COUNT);
        if (dispatch_idx >= alloc_count * IRCACHE_VALIDATION_SAMPLES_PER_FRAME) {
            return;
        }
    #endif

    const uint entry_idx = ircache_entry_indirection_buf[dispatch_idx / IRCACHE_VALIDATION_SAMPLES_PER_FRAME];
    const uint sample_idx = dispatch_idx % IRCACHE_VALIDATION_SAMPLES_PER_FRAME;
    const uint life = ircache_life_buf.Load(entry_idx * 4);

    DiffuseBrdf brdf;
    brdf.albedo = 1.0.xxx;

    const SampleParams sample_params = SampleParams::from_spf_entry_sample_frame(
        IRCACHE_VALIDATION_SAMPLES_PER_FRAME,
        entry_idx,
        sample_idx,
        frame_constants.frame_index);

    const uint octa_idx = sample_params.octa_idx();
    const uint output_idx = entry_idx * IRCACHE_AUX_STRIDE + octa_idx;

    float invalidity = 0;

    {
        // TODO: wz not used. slim down.
        Reservoir1spp r = Reservoir1spp::from_raw(asuint(ircache_aux_buf[output_idx].xy));

        if (r.M > 0) {
            float4 prev_value_and_count =
                ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2]
                * float4((frame_constants.pre_exposure_delta).xxx, 1);

            Vertex prev_entry = unpack_vertex(VertexPacked(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2 * 2]));

            // Validate the previous sample
            IrcacheTraceResult prev_traced = ircache_trace(prev_entry, brdf, SampleParams::from_raw(r.payload), life);

            const float prev_self_lighting_limiter = 
                select(USE_SELF_LIGHTING_LIMITER
                , lerp(0.5, 1, smoothstep(-0.1, 0, dot(prev_traced.direction, prev_entry.normal)))
                , 1.0);

            const float3 a = prev_traced.incident_radiance * prev_self_lighting_limiter;
            const float3 b = prev_value_and_count.rgb;
            const float3 dist3 = abs(a - b) / (a + b);
            const float dist = max(dist3.r, max(dist3.g, dist3.b));
            invalidity = smoothstep(0.1, 0.5, dist);
            r.M = max(0, min(r.M, exp2(log2(float(IRCACHE_RESTIR_M_CLAMP)) * (1.0 - invalidity))));

            // Update the stored value too.
            // TODO: try the update heuristics from the diffuse trace
            prev_value_and_count.rgb = a;

            ircache_aux_buf[output_idx].xy = asfloat(r.as_raw());
            ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2] = prev_value_and_count;
        }
    }

    // Also reduce M of the neighbors in case we have fewer validation rays than irradiance rays.
    #if 1
        if (IRCACHE_VALIDATION_SAMPLES_PER_FRAME < IRCACHE_SAMPLES_PER_FRAME) {
            if (invalidity > 0) {
                const uint PERIOD = IRCACHE_OCTA_DIMS2 / IRCACHE_VALIDATION_SAMPLES_PER_FRAME;
                const uint OTHER_PERIOD = IRCACHE_OCTA_DIMS2 / IRCACHE_SAMPLES_PER_FRAME;

                for (uint xor = OTHER_PERIOD; xor < PERIOD; xor *= 2) {
                    const uint idx = output_idx ^ xor;
                    Reservoir1spp r = Reservoir1spp::from_raw(asuint(ircache_aux_buf[idx].xy));
                    r.M = max(0, min(r.M, exp2(log2(float(IRCACHE_RESTIR_M_CLAMP)) * (1.0 - invalidity))));
                    ircache_aux_buf[idx].xy = asfloat(r.as_raw());
                }
            }
        }
    #endif
}
