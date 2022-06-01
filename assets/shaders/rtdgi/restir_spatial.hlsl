#include "../inc/color.hlsl"
#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/reservoir.hlsl"
#include "rtdgi_restir_settings.hlsl"
#include "rtdgi_common.hlsl"

[[vk::binding(0)]] Texture2D<uint2> reservoir_input_tex;
[[vk::binding(1)]] Texture2D<float3> radiance_input_tex;
[[vk::binding(2)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(3)]] Texture2D<float> half_depth_tex;
[[vk::binding(4)]] Texture2D<float> depth_tex;
[[vk::binding(5)]] Texture2D<float> half_ssao_tex;
[[vk::binding(6)]] Texture2D<uint4> temporal_reservoir_packed_tex;
[[vk::binding(7)]] Texture2D<float4> reprojected_gi_tex;
[[vk::binding(8)]] RWTexture2D<uint2> reservoir_output_tex;
[[vk::binding(9)]] RWTexture2D<float3> radiance_output_tex;
[[vk::binding(10)]] cbuffer _ {
    float4 gbuffer_tex_size;
    float4 output_tex_size;
    uint spatial_reuse_pass_idx;
};

#define USE_SSAO_WEIGHING 1
#define ALLOW_REUSE_OF_BACKFACING 1

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

// Two-thirds of SmeLU
float normal_inluence_nonlinearity(float x, float b) {
    return x < -b
        ? 0
        : (x + b) * (x + b) / (4 * b);
}

float distanceSquared(float2 a, float2 b) {
	a -= b;
	return dot(a, a);
}

void swap(inout float a, inout float b) {
    float temp = a;
    a = b;
    b = temp;
}

/**
   \param csOrigin Camera-space ray origin, which must be
   within the view volume and must have z < -0.01 and project within the valid screen rectangle
   \param csDirection Unit length camera-space ray direction
   \param projectToPixelMatrix A projection matrix that maps to pixel coordinates (not [-1, +1] normalized device coordinates)
   \param csZBuffer The depth or camera-space Z buffer, depending on the value of \a csZBufferIsHyperbolic
   \param csZBufferSize Dimensions of csZBuffer
   \param relativeThickness Camera space thickness to ascribe to each pixel in the depth buffer
   \param csZBufferIsHyperbolic True if csZBuffer is an OpenGL depth buffer, false (faster) if
   csZBuffer contains (negative) "linear" camera space z values. Const so that the compiler can evaluate the branch based on it at compile time
   \param clipInfo See G3D::Camera documentation
   \param nearPlaneZ Negative number
   \param stride Step in horizontal or vertical pixels between samples. This is a float
   because integer math is slow on GPUs, but should be set to an integer >= 1
   \param jitterFraction  Number between 0 and 1 for how far to bump the ray in stride units
   to conceal banding artifacts
   \param maxSteps Maximum number of iterations. Higher gives better images but may be slow
   \param maxRayTraceDistance Maximum camera-space distance to trace before returning a miss
   \param hitPixel Pixel coordinates of the first intersection with the scene
   \param csHitvec Camera space location of the ray hit
   Single-layer
*/
bool traceScreenSpaceRay1b(float3            csOrigin,
                          float3            csDirection,
                          float4x4            projectToPixelMatrix,
                          float2          csZBufferSize,
                          float           relativeThickness,
                          float           nearPlaneZ,
                          float           stride,
                          float           jitterFraction,
                          float           maxSteps,
                          float        maxRayTraceDistance,
                          out float2        hitPixel,
                          out float3        csHitvec) {

    // Clip ray to a near plane in 3D (doesn't have to be *the* near plane, although that would be a good idea)
    float rayLength = ((csOrigin.z + csDirection.z * maxRayTraceDistance) > nearPlaneZ) ?
        (nearPlaneZ - csOrigin.z) / csDirection.z :
        maxRayTraceDistance;
    float3 csEndPoint = csDirection * rayLength + csOrigin;

    // Project into screen space
    //float4 H0 = mul(projectToPixelMatrix * float4(csOrigin, 1.0);
    //float4 H1 = projectToPixelMatrix * float4(csEndPoint, 1.0);

	float4 H0 = (mul(projectToPixelMatrix, float4(csOrigin, 1.0)));
	float4 H1 = (mul(projectToPixelMatrix, float4(csEndPoint, 1.0)));

    // There are a lot of divisions by w that can be turned into multiplications
    // at some minor precision loss...and we need to interpolate these 1/w values
    // anyway.
    //
    // Because the caller was required to clip to the near plane,
    // this homogeneous division (projecting from 4D to 2D) is guaranteed
    // to succeed.
    float k0 = 1.0 / H0.w;
    float k1 = 1.0 / H1.w;

    // Switch the original vecs to values that interpolate linearly in 2D
    float3 Q0 = csOrigin * k0;
    float3 Q1 = csEndPoint * k1;

    // Screen-space endvecs
    float2 P0 = H0.xy * k0;
    float2 P1 = H1.xy * k1;

    P0 = cs_to_uv(P0) * gbuffer_tex_size.xy;
    P1 = cs_to_uv(P1) * gbuffer_tex_size.xy;

    // [Optional clipping to frustum sides here]

    // Initialize to off screen
    hitPixel = float2(-1.0, -1.0);


    // If the line is degenerate, make it cover at least one pixel
    // to avoid handling zero-pixel extent as a special case later
    P1 += (distanceSquared(P0, P1) < 0.0001) ? 0.01 : 0.0;

    float2 delta = P1 - P0;

    // Permute so that the primary iteration is in x to reduce
    // large branches later
    bool permute = false;
    if (abs(delta.x) < abs(delta.y)) {
        // More-vertical line. Create a permutation that swaps x and y in the output
        permute = true;

        // Directly swizzle the inputs
        delta = delta.yx;
        P1 = P1.yx;
        P0 = P0.yx;
    }

    // From now on, "x" is the primary iteration direction and "y" is the secondary one

    float stepDirection = delta.x > 0 ? 1 : -1;
    float invdx = stepDirection / delta.x;
    float2 dP = float2(stepDirection, invdx * delta.y);

    // Track the derivatives of Q and k
    float3 dQ = (Q1 - Q0) * invdx;
    float   dk = (k1 - k0) * invdx;

    // Scale derivatives by the desired pixel stride
    dP *= stride; dQ *= stride; dk *= stride;

    // Offset the starting values by the jitter fraction
    P0 += dP * jitterFraction; Q0 += dQ * jitterFraction; k0 += dk * jitterFraction;

    // Slide P from P0 to P1, (now-homogeneous) Q from Q0 to Q1, and k from k0 to k1
    float3 Q = Q0;
    float  k = k0;

    // We track the ray depth at +/- 1/2 pixel to treat pixels as clip-space solid
    // voxels. Because the depth at -1/2 for a given pixel will be the same as at
    // +1/2 for the previous iteration, we actually only have to compute one value
    // per iteration.
    float prevZMaxEstimate = csOrigin.z;
    float stepCount = 0.0;
    float rayZMax = prevZMaxEstimate, rayZMin = prevZMaxEstimate;
    float sceneZMax = rayZMax + 1e4;

    // P1.x is never modified after this vec, so pre-scale it by
    // the step direction for a signed comparison
    float end = P1.x * stepDirection;

    // We only advance the z field of Q in the inner loop, since
    // Q.xy is never used until after the loop terminates.

    float2 P = P0;
    for (int kStep= 0; kStep < 100; kStep++) {
        #if 1
        if (!(((P.x * stepDirection) <= end) &&
             (stepCount < maxSteps) &&
             ((rayZMax < sceneZMax * relativeThickness) ||
              (rayZMin > sceneZMax * 1.002)) &&
              (sceneZMax != 0.0)))
            break;
        #endif

        hitPixel = permute ? P.yx : P;
        //hitPixel.y = csZBufferSize.y - hitPixel.y;

        // The depth range that the ray covers within this loop
        // iteration.  Assume that the ray is moving in increasing z
        // and swap if backwards.  Because one end of the interval is
        // shared between adjacent iterations, we track the previous
        // value and then swap as needed to ensure correct ordering
        rayZMin = prevZMaxEstimate;

        // Compute the value at 1/2 pixel into the future
        rayZMax = (dQ.z * 0.5 + Q.z) / (dk * 0.5 + k);
        prevZMaxEstimate = rayZMax;
        if (rayZMin > rayZMax) { swap(rayZMin, rayZMax); }

        // Camera-space z of the background
        sceneZMax = depth_to_view_z(depth_tex[int2(hitPixel)]);

        // depth buffer basic 0.1 hyperbolic opengl itis.
        //sceneZMax = reconstructCSZ(sceneZMax, clipInfo);

        P += dP; Q.z += dQ.z; k += dk; stepCount += 1.0;

    } // pixel on ray


    Q.xy += dQ.xy * stepCount;
    csHitvec = Q * (1.0 / k);

    // Matches the new loop condition:
    return (rayZMax >= sceneZMax * relativeThickness) && (rayZMin <= sceneZMax);
}

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID) {
    const uint2 hi_px = px * 2 + HALFRES_SUBSAMPLE_OFFSET;
    float depth = half_depth_tex[px];

    const uint seed = frame_constants.frame_index + spatial_reuse_pass_idx * 123;
    uint rng = hash3(uint3(px, seed));

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    const float3 center_normal_vs = half_view_normal_tex[px].rgb;
    const float3 center_normal_ws = direction_view_to_world(center_normal_vs);
    const float center_depth = half_depth_tex[px];
    const float center_ssao = half_ssao_tex[px].r;

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState::create();
    Reservoir1spp reservoir = Reservoir1spp::create();

    float3 dir_sel = 1;

    float sample_radius_offset = uint_to_u01_float(hash1_mut(rng));

    Reservoir1spp center_r = Reservoir1spp::from_raw(reservoir_input_tex[px]);

    // TODO: drive this via variance, shrink when it's low. 80 is a bit of a blur...
    float kernel_tightness = 1.0 - center_ssao;

    const uint SAMPLE_COUNT_PASS0 = 8;
    const uint SAMPLE_COUNT_PASS1 = 5;

    const float MAX_INPUT_M_IN_PASS0 = RESTIR_TEMPORAL_M_CLAMP;
    const float MAX_INPUT_M_IN_PASS1 = MAX_INPUT_M_IN_PASS0 * SAMPLE_COUNT_PASS0;
    const float MAX_INPUT_M_IN_PASS = spatial_reuse_pass_idx == 0 ? MAX_INPUT_M_IN_PASS0 : MAX_INPUT_M_IN_PASS1;

    // TODO: unify with `kernel_tightness`
    if (RTDGI_RESTIR_SPATIAL_USE_KERNEL_NARROWING) {
        kernel_tightness = lerp(
            kernel_tightness, 1.0,
            0.5 * smoothstep(MAX_INPUT_M_IN_PASS * 0.5, MAX_INPUT_M_IN_PASS, center_r.M));
    }

    float max_kernel_radius =
        spatial_reuse_pass_idx == 0
        ? lerp(32.0, 12.0, kernel_tightness)
        : lerp(16.0, 6.0, kernel_tightness);

    // TODO: only run more passes where absolutely necessary
    if (spatial_reuse_pass_idx >= 2) {
        max_kernel_radius = 8;
    }

    const float2 dist_to_edge_xy = min(float2(px), output_tex_size.xy - px);
    const float allow_edge_overstep = center_r.M < 10 ? 100.0 : 1.25;
    //const float allow_edge_overstep = 1.25;
    const float2 kernel_radius = min(max_kernel_radius, dist_to_edge_xy * allow_edge_overstep);
    //const float2 kernel_radius = max_kernel_radius;

    uint sample_count = DIFFUSE_GI_USE_RESTIR
        ? (spatial_reuse_pass_idx == 0 ? SAMPLE_COUNT_PASS0 : SAMPLE_COUNT_PASS1)
        : 1;

    #if 1
        // Scrambling angles here would be nice, but results in bad cache thrashing.
        // Quantizing the offsets results in mild cache abuse, and fixes most of the artifacts
        // (flickering near edges, e.g. under sofa in the UE5 archviz apartment scene).
        const uint2 ang_offset_seed = spatial_reuse_pass_idx == 0
            ? (px >> 3)
            : (px >> 2);
    #else
        // Haha, cache go brrrrrrr.
        const uint2 ang_offset_seed = px;
    #endif

    float ang_offset = uint_to_u01_float(hash3(
        uint3(ang_offset_seed, frame_constants.frame_index * 2 + spatial_reuse_pass_idx)
    )) * M_PI * 2;

    if (!RESTIR_USE_SPATIAL) {
        sample_count = 1;
    }

    float3 radiance_output = 0;

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        //float ang = M_PI / 2;
        float ang = (sample_i + ang_offset) * GOLDEN_ANGLE;
        float2 radius =
            0 == sample_i
            ? 0
            : (pow(float(sample_i + sample_radius_offset) / sample_count, 0.5) * kernel_radius);
        int2 rpx_offset = float2(cos(ang), sin(ang)) * radius;

        const bool is_center_sample = sample_i == 0;
        //const bool is_center_sample = all(rpx_offset == 0);

        const int2 rpx = px + rpx_offset;

        const uint2 reservoir_raw = reservoir_input_tex[rpx];
        if (0 == reservoir_raw.x) {
            // Invalid reprojectoin
            continue;
        }

        Reservoir1spp r = Reservoir1spp::from_raw(reservoir_raw);

        r.M = min(r.M, 500);

        const uint2 spx = reservoir_payload_to_px(r.payload);

        const TemporalReservoirOutput spx_packed = TemporalReservoirOutput::from_raw(temporal_reservoir_packed_tex[spx]);

        // TODO: to recover tiny highlights, consider raymarching first, and then using the screen-space
        // irradiance value instead of this.
        const float reused_luminance = spx_packed.luminance;

        float visibility = 1;
        float relevance = 1;

        // Note: we're using `rpx` (neighbor reservoir px) here instead of `spx` (original ray px),
        // since we're merging with the stream of the neighbor and not the original ray.
        //
        // The distinction is in jacobians -- during every exchange, they get adjusted so that the target
        // pixel has correctly distributed rays. If we were to merge with the original pixel's stream,
        // we'd be applying the reservoirs several times.
        //
        // Consider for example merging a pixel with itself (no offset) multiple times over; we want
        // the jacobian to be 1.0 in that case, and not to reflect wherever its ray originally came from.

        const int2 sample_offset = int2(px) - int2(rpx);
        const float sample_dist2 = dot(sample_offset, sample_offset);
        const float3 sample_normal_vs = half_view_normal_tex[rpx].rgb;

        float3 sample_radiance;
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
            sample_radiance = radiance_input_tex[rpx];
        }

        const float normal_cutoff = 0.1;

        const float normal_similarity_dot = dot(sample_normal_vs, center_normal_vs);
        #if ALLOW_REUSE_OF_BACKFACING
            // Allow reuse even with surfaces that face away, but weigh them down.
            relevance *= normal_inluence_nonlinearity(normal_similarity_dot, 0.5)
                / normal_inluence_nonlinearity(1.0, 0.5);
        #else
            relevance *= max(0, normal_similarity_dot);
        #endif

        const float sample_ssao = half_ssao_tex[rpx];

        #if USE_SSAO_WEIGHING
            relevance *= 1 - abs(sample_ssao - center_ssao);
        #endif

        const float2 rpx_uv = get_uv(
            rpx * 2 + HALFRES_SUBSAMPLE_OFFSET,
            gbuffer_tex_size);
        const float rpx_depth = half_depth_tex[rpx];
        
        if (rpx_depth == 0.0) {
            continue;
        }

        const ViewRayContext rpx_ray_ctx = ViewRayContext::from_uv_and_depth(rpx_uv, rpx_depth);

        const float2 spx_uv = get_uv(
            spx * 2 + HALFRES_SUBSAMPLE_OFFSET,
            gbuffer_tex_size);
        const ViewRayContext spx_ray_ctx = ViewRayContext::from_uv_and_depth(spx_uv, spx_packed.depth);
        const float3 sample_hit_ws = spx_packed.ray_hit_offset_ws + spx_ray_ctx.ray_hit_ws();

        const float3 reused_dir_to_sample_hit_unnorm_ws = sample_hit_ws - rpx_ray_ctx.ray_hit_ws();

        //const float reused_luminance = sample_hit_ws_and_luminance.a;

        // Note: we want the neighbor's sample, which might have been resampled already.
        const float reused_dist = length(reused_dir_to_sample_hit_unnorm_ws);
        const float3 reused_dir_to_sample_hit_ws = reused_dir_to_sample_hit_unnorm_ws / reused_dist;

        const float3 dir_to_sample_hit_unnorm = sample_hit_ws - view_ray_context.ray_hit_ws();
        const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
        const float3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);

        // Reject hits below the normal plane
        const bool is_below_normal_plane = dot(dir_to_sample_hit, center_normal_ws) < 1e-5;

        if ((!is_center_sample && is_below_normal_plane) || !(relevance > 0)) {
            continue;
        }

        // Reject neighbors with vastly different depths
        if (!is_center_sample) {
            // Clamp the normal_vs.z so that we don't get arbitrarily loose depth comparison at grazing angles.
            const float depth_diff = abs(max(0.3, center_normal_vs.z) * (center_depth / rpx_depth - 1.0));

            const float depth_threshold =
                spatial_reuse_pass_idx == 0
                ? 0.15
                : 0.1;

            relevance *= 1 - smoothstep(0.0, depth_threshold, depth_diff);
        }

        const float USE_DDA = !true;

        if (USE_DDA) {
            const float2 ray_orig_uv = spx_uv;
    		//const float surface_offset_len = length(spx_ray_ctx.ray_hit_vs() - view_ray_context.ray_hit_vs());
            const float surface_offset_len = length(
                // Use the center depth for simplicity; this doesn't need to be exact.
                // Faster, looks about the same.
                ViewRayContext::from_uv_and_depth(ray_orig_uv, depth).ray_hit_vs() - view_ray_context.ray_hit_vs()
            );

            // TODO: finish the derivations, don't perspective-project for every sample.

            // Multiplier over the surface offset from the center to the neighbor
            const float MAX_RAYMARCH_DIST_MULT = 2.0;

            // Trace towards the hit point.

            const float3 raymarch_start_ws = view_ray_context.ray_hit_ws();
            const float3 raymarch_dir_unnorm_ws = sample_hit_ws - raymarch_start_ws;
            const float3 raymarch_end_ws =
                raymarch_start_ws
                // TODO: what's a good max distance to raymarch? Probably need to project some stuff
                + raymarch_dir_unnorm_ws * min(1.0, MAX_RAYMARCH_DIST_MULT * surface_offset_len / length(raymarch_dir_unnorm_ws));

            const float2 raymarch_end_uv = cs_to_uv(position_world_to_clip(raymarch_end_ws).xy);
            const float2 raymarch_uv_delta = raymarch_end_uv - uv;
            const float2 raymarch_len_px = raymarch_uv_delta * output_tex_size.xy;

            //float3 raymarch_end_ws = raymarch_start_ws + float3(0, 1, 0);

            const float3 raymarch_start_vs = position_world_to_view(raymarch_start_ws);
            const float3 raymarch_end_vs = position_world_to_view(raymarch_end_ws);
            const float3 raymarch_offset_vs = raymarch_end_vs - raymarch_start_vs;


        	float2 hitPixel;
        	float3 hitPoint;

            const float stride = floor(max(1, length(raymarch_len_px) / 4));
            const bool hit = traceScreenSpaceRay1b(
                raymarch_start_vs,
                normalize(raymarch_offset_vs),
                frame_constants.view_constants.view_to_sample,
                output_tex_size.xy,
                1.04,   //           relativeThickness,
                0.0,   //           nearPlaneZ,
                stride,  //           stride,
                0.5, //           jitterFraction,
                4, //           maxSteps,
                length(raymarch_offset_vs),   //        maxRayTraceDistance,
                hitPixel,
                hitPoint);


            if (hit) {
                visibility = 0;
            }
        }

        if (!USE_DDA) {
            // Raymarch to check occlusion
            if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH && !is_center_sample) {
                const float2 ray_orig_uv = spx_uv;

        		//const float surface_offset_len = length(spx_ray_ctx.ray_hit_vs() - view_ray_context.ray_hit_vs());
                const float surface_offset_len = length(
                    // Use the center depth for simplicity; this doesn't need to be exact.
                    // Faster, looks about the same.
                    ViewRayContext::from_uv_and_depth(ray_orig_uv, depth).ray_hit_vs() - view_ray_context.ray_hit_vs()
                );

                // TODO: finish the derivations, don't perspective-project for every sample.

                // Multiplier over the surface offset from the center to the neighbor
                const float MAX_RAYMARCH_DIST_MULT = 2.0;

                // Trace towards the hit point.

                const float3 raymarch_dir_unnorm_ws = sample_hit_ws - view_ray_context.ray_hit_ws();
                const float3 raymarch_end_ws =
                    view_ray_context.ray_hit_ws()
                    // TODO: what's a good max distance to raymarch? Probably need to project some stuff
                    + raymarch_dir_unnorm_ws * min(1.0, MAX_RAYMARCH_DIST_MULT * surface_offset_len / length(raymarch_dir_unnorm_ws));

                const float2 raymarch_end_uv = cs_to_uv(position_world_to_clip(raymarch_end_ws).xy);
                const float2 raymarch_uv_delta = raymarch_end_uv - uv;
                const float2 raymarch_len_px = raymarch_uv_delta * output_tex_size.xy;

                const uint MIN_PX_PER_STEP = 2;
                const uint MAX_TAPS = 4;

                const int k_count = min(MAX_TAPS, int(floor(length(raymarch_len_px) / MIN_PX_PER_STEP)));

                // Depth values only have the front; assume a certain thickness.
                const float Z_LAYER_THICKNESS = 0.05;

                const float3 raymarch_start_cs = view_ray_context.ray_hit_cs.xyz;
                const float3 raymarch_end_cs = position_world_to_clip(raymarch_end_ws).xyz;
                const float depth_step_per_px = (raymarch_end_cs.z - raymarch_start_cs.z) / length(raymarch_len_px);
                const float depth_step_per_z = (raymarch_end_cs.z - raymarch_start_cs.z) / length(raymarch_end_cs.xy - raymarch_start_cs.xy);

                float t_step = 1.0 / k_count;
                float t = 0.5 * t_step;
                for (int k = 0; k < k_count; ++k) {
                    const float3 interp_pos_cs = lerp(raymarch_start_cs, raymarch_end_cs, t);

                    // The point-sampled UV could end up with a quite different depth value
                    // than the one interpolated along the ray (which is not quantized).
                    // This finds a conservative bias for the comparison.
                    const float2 uv_at_interp = cs_to_uv(interp_pos_cs.xy);

#if 0
                    const uint2 px_at_interp = floor(uv_at_interp * output_tex_size.xy);
                    const float depth_at_interp = half_depth_tex[px_at_interp];
                    const float2 quantized_cs_at_interp = uv_to_cs((px_at_interp + 0.5) / output_tex_size.xy);
#elif 0
                    const uint2 px_at_interp = floor(uv_at_interp * gbuffer_tex_size.xy);
                    const float depth_at_interp = depth_tex[px_at_interp];
                    const float2 quantized_cs_at_interp = uv_to_cs((px_at_interp + 0.5) / gbuffer_tex_size.xy);
#else
                    const uint2 px_at_interp = (uint2(floor(uv_at_interp * gbuffer_tex_size.xy)) & ~1u) + HALFRES_SUBSAMPLE_OFFSET;
                    const float depth_at_interp = half_depth_tex[px_at_interp >> 1u];
                    const float2 quantized_cs_at_interp = uv_to_cs((px_at_interp + 0.25 + HALFRES_SUBSAMPLE_OFFSET * 0.5) / gbuffer_tex_size.xy);
#endif

                    const float biased_interp_z = raymarch_start_cs.z + depth_step_per_z * length(quantized_cs_at_interp - raymarch_start_cs.xy);

                    // TODO: get this const as low as possible to get micro-shadowing
                    //if (depth_at_interp > interp_pos_cs.z)
                    //if (depth_at_interp > interp_pos_cs.z * 1.003)
                    if (depth_at_interp > biased_interp_z)
                    {
                        const float depth_diff = inverse_depth_relative_diff(interp_pos_cs.z, depth_at_interp);

                        // TODO, BUG: if the hit surface is emissive, this ends up casting a shadow from it,
                        // without taking the emission into consideration.

                        float hit = smoothstep(
                            Z_LAYER_THICKNESS,
                            Z_LAYER_THICKNESS * 0.5,
                            depth_diff);

                        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
                            const float3 hit_radiance = reprojected_gi_tex.SampleLevel(sampler_llc, cs_to_uv(interp_pos_cs.xy), 0).rgb;
                            const float hit_luminance = sRGB_to_luminance(hit_radiance);

                            sample_radiance = lerp(sample_radiance, hit_radiance, hit);

                            // Heuristic: don't allow getting _brighter_ from accidental
                            // hits reused from neighbors. This can cause some darkening,
                            // but also fixes reduces noise (expecting to hit dark, hitting bright),
                            // and improves a few cases that otherwise look unshadowed.
                            visibility *= min(1.0, reused_luminance / hit_luminance);
                        } else {
                            visibility *= 1 - hit;
                        }

                        if (depth_diff > Z_LAYER_THICKNESS) {
                            // Going behind an object; could be sketchy.
                            // Note: maybe nuke.. causes bias around foreground objects.
                            //relevance *= 0.2;
                        }
                    }

                    t += t_step;
                }
    		}
        }

        const float3 sample_hit_normal_ws = spx_packed.hit_normal_ws;

        // phi_2^r in the ReSTIR GI paper
        const float center_to_hit_vis = -dot(sample_hit_normal_ws, dir_to_sample_hit);

        // phi_2^q
        const float reused_to_hit_vis = -dot(sample_hit_normal_ws, reused_dir_to_sample_hit_ws);

        float p_q = 1;
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
            p_q *= sRGB_to_luminance(sample_radiance);
        } else {
            p_q *= reused_luminance;
        }

        // Unlike in temporal reuse, here we can (and should) be running this.
        p_q *= max(0, dot(dir_to_sample_hit, center_normal_ws));

        float jacobian = 1;

        // Distance falloff. Needed to avoid leaks.
        jacobian *= reused_dist / dist_to_sample_hit;
        jacobian *= jacobian;

        // N of hit dot -L. Needed to avoid leaks. Without it, light "hugs" corners.
        //
        // Note: importantly, using the neighbor's data, not the original ray.
        jacobian *= clamp(center_to_hit_vis / reused_to_hit_vis, 0, 1e4);

        // Clearly wrong, but!:
        // The Jacobian introduces additional noise in corners, which is difficult to filter.
        // We still need something _resembling_ the jacobian in order to get directional cutoff,
        // and avoid leaks behind surfaces, but we don't actually need the precise Jacobian.
        // This causes us to lose some energy very close to corners, but with the near field split,
        // we don't need it anyway -- and it's better not to have the larger dark halos near corners,
        // which fhe full jacobian can cause due to imperfect integration (color bbox filters, etc).
        jacobian = sqrt(jacobian);

        if (is_center_sample) {
            jacobian = 1;
        }

        // Clamp neighbors give us a hit point that's considerably easier to sample
        // from our own position than from the neighbor. This can cause some darkening,
        // but prevents fireflies.
        //
        // The darkening occurs in corners, where micro-bounce should be happening instead.

        if (RTDGI_RESTIR_USE_JACOBIAN_BASED_REJECTION) {
            #if 1
                // Doesn't over-darken corners as much
                jacobian = min(jacobian, RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE);
            #else
                // Slightly less noise
                if (jacobian > RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE) { continue; }
            #endif
        }

        if (!(p_q >= 0)) {
            continue;
        }

        r.M *= relevance;

        if (reservoir.update_with_stream(
            r, p_q, visibility * jacobian,
            stream_state, r.payload, rng
        )) {
            dir_sel = dir_to_sample_hit;
            radiance_output = sample_radiance;
        }
    }

    reservoir.finish_stream(stream_state);
    reservoir.W = min(reservoir.W, RESTIR_RESERVOIR_W_CLAMP);

    reservoir_output_tex[px] = reservoir.as_raw();

    if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
        radiance_output_tex[px] = radiance_output;
    }
}
