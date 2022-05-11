// Trace rays between the previous ReSTIR ircache trace origins and the newly proposed ones,
// reducing the memory of reservoirs that are inaccessible now.
//
// This speeds up transitions between indoors/outdoors for cache entries which span both sides.

#include "../inc/frame_constants.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/reservoir.hlsl"
#include "../inc/mesh.hlsl"
#include "ircache_constants.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] StructuredBuffer<VertexPacked> ircache_spatial_buf;
[[vk::binding(1)]] StructuredBuffer<uint> ircache_life_buf;
[[vk::binding(2)]] RWStructuredBuffer<VertexPacked> ircache_reposition_proposal_buf;
[[vk::binding(3)]] ByteAddressBuffer ircache_meta_buf;
[[vk::binding(4)]] RWStructuredBuffer<float4> ircache_aux_buf;
[[vk::binding(5)]] StructuredBuffer<uint> ircache_entry_indirection_buf;

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
        if (dispatch_idx >= alloc_count * IRCACHE_OCTA_DIMS2) {
            return;
        }
    #endif

    const uint entry_idx = ircache_entry_indirection_buf[dispatch_idx / IRCACHE_OCTA_DIMS2];
    const uint octa_idx = dispatch_idx % IRCACHE_OCTA_DIMS2;
    const uint life = ircache_life_buf[entry_idx];

    if (!is_ircache_entry_life_valid(life)) {
        return;
    }

    const Vertex entry = unpack_vertex(ircache_spatial_buf[entry_idx]);

    const uint output_idx = entry_idx * IRCACHE_AUX_STRIDE + octa_idx;

    Reservoir1spp r = Reservoir1spp::from_raw(asuint(ircache_aux_buf[output_idx].xy));
    Vertex prev_entry = unpack_vertex(VertexPacked(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2 * 2]));

    // Reduce weight of samples whose trace origins are not accessible now
    if (rt_is_shadowed(
        acceleration_structure,
        new_ray(
            entry.position,
            prev_entry.position - entry.position,
            0.001,
            0.999
    ))) {
        r.M *= 0.8;
        ircache_aux_buf[output_idx].xy = asfloat(r.as_raw());
    }
}
