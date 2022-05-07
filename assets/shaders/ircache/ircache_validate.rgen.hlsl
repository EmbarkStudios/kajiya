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
[[vk::binding(8)]] RWStructuredBuffer<float4> ircache_irradiance_buf;
[[vk::binding(9)]] RWStructuredBuffer<float4> ircache_aux_buf;
[[vk::binding(10)]] RWStructuredBuffer<uint> ircache_pool_buf;
[[vk::binding(11)]] StructuredBuffer<uint> ircache_entry_indirection_buf;
[[vk::binding(12)]] RWStructuredBuffer<uint> ircache_entry_cell_buf;

#include "../inc/sun.hlsl"
#include "../wrc/lookup.hlsl"

#include "ircache_trace_common.inc.hlsl"

[shader("raygeneration")]
void main() {
    if (IRCACHE_FREEZE) {
        return;
    }

    const uint dispatch_idx = DispatchRaysIndex().x;
    const uint entry_idx = ircache_entry_indirection_buf[dispatch_idx / IRCACHE_SAMPLES_PER_FRAME];
    const uint sample_idx = dispatch_idx % IRCACHE_SAMPLES_PER_FRAME;
    const uint life = ircache_life_buf.Load(entry_idx * 4);
    const uint rank = ircache_entry_life_to_rank(life);

    VertexPacked packed_entry = ircache_spatial_buf[entry_idx];
    const Vertex entry = unpack_vertex(packed_entry);

    DiffuseBrdf brdf;
    //const float3x3 tangent_to_world = build_orthonormal_basis(entry.normal);

    brdf.albedo = 1.0.xxx;

    const SampleParams sample_params = SampleParams::from_entry_sample_frame(entry_idx, sample_idx, frame_constants.frame_index);
    const uint octa_idx = sample_params.octa_idx();
    const uint output_idx = entry_idx * IRCACHE_AUX_STRIDE + octa_idx;

    Reservoir1spp r = Reservoir1spp::from_raw(ircache_aux_buf[output_idx]);

    const uint M_CLAMP = 30;

    if (r.M > 0) {
        float4 prev_value_and_count =
            ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2]
            * float4((frame_constants.pre_exposure_delta).xxx, 1);

        Vertex prev_entry = unpack_vertex(VertexPacked(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2 * 2]));

        // Validate the previous sample
        IrcacheTraceResult prev_traced = ircache_trace(prev_entry, brdf, SampleParams::from_raw(r.payload), life);

        const float prev_self_lighting_limiter = 
            USE_SELF_LIGHTING_LIMITER
            ? lerp(0.75, 1, smoothstep(-0.1, 0, dot(prev_traced.direction, prev_entry.normal)))
            : 1.0;

        const float3 a = prev_traced.incident_radiance * prev_self_lighting_limiter;
        const float3 b = prev_value_and_count.rgb;
        const float3 dist3 = abs(a - b) / (a + b);
        const float dist = max(dist3.r, max(dist3.g, dist3.b));
        const float invalidity = smoothstep(0.1, 0.5, dist);
        r.M = max(0, min(r.M, exp2(log2(float(M_CLAMP)) * (1.0 - invalidity))));

        // Update the stored value too.
        // TODO: Feels like the W might need to be updated too, because we would
        // have picked this sample with a different probability...
        prev_value_and_count.rgb = a;

        ircache_aux_buf[output_idx] = r.as_raw();
        ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2] = prev_value_and_count;
    }
}
