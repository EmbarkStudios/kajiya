#include "../inc/frame_constants.hlsl"

#define USE_RTDGI_CONTROL_VARIATES 1

// Must match CPU code. Seach token: d4109bba-438f-425e-8667-19e591be9a56
static const uint CSGI_VOLUME_DIMS = 64;
#define CSGI_CASCADE_COUNT 1

static const uint CSGI_CARDINAL_DIRECTION_COUNT = 6;
static const uint CSGI_CARDINAL_SUBRAY_COUNT = 5;

static const uint CSGI_DIAGONAL_DIRECTION_COUNT = 8;
static const uint CSGI_DIAGONAL_SUBRAY_COUNT = 3;

static const uint CSGI_TOTAL_SUBRAY_COUNT =
    CSGI_CARDINAL_DIRECTION_COUNT * CSGI_CARDINAL_SUBRAY_COUNT
    + CSGI_DIAGONAL_DIRECTION_COUNT * CSGI_DIAGONAL_SUBRAY_COUNT;


// X coord in the image of the first subray for diagonal directions.
// Basically after all the cardinal subrays.
static const uint CSGI_DIAGONAL_DIRECTION_SUBRAY_OFFSET =
    CSGI_VOLUME_DIMS * CSGI_CARDINAL_DIRECTION_COUNT * CSGI_CARDINAL_SUBRAY_COUNT;

static const uint CSGI_TOTAL_DIRECTION_COUNT = CSGI_CARDINAL_DIRECTION_COUNT + CSGI_DIAGONAL_DIRECTION_COUNT;

#define CSGI_SUBRAY_PACKING_AS_SUBVOLUMES 0

#define CSGI_ACCUM_HYSTERESIS 0.25
//#define CSGI_ACCUM_HYSTERESIS 1.0

// Note: the "csgi subray combine" pass must be enabled if this is 0.
#define CSGI_SUBRAY_COMBINE_DURING_SWEEP 1

float csgi_voxel_size(uint cascade_idx) {
    return frame_constants.gi_cascades[cascade_idx].voxel_size;
}

float csgi_volume_size(uint cascade_idx) {
    return frame_constants.gi_cascades[cascade_idx].volume_size;
}

// The grid is shifted, so the _center_ and not _origin_ of cascade 0 is at origin.
// This moves the seams in the origin-centered grid away from origin.
// Must match CPU code. Search token: 3e7ddeec-afbb-44e4-8b75-b54276c6df2b
#define CSGI_VOLUME_ORIGIN ((-float(CSGI_VOLUME_DIMS) / 2.0) * csgi_voxel_size(0))

static const int3 CSGI_DIRECT_DIRS[CSGI_CARDINAL_DIRECTION_COUNT] = {
    int3(-1, 0, 0),
    int3(1, 0, 0),
    int3(0, -1, 0),
    int3(0, 1, 0),
    int3(0, 0, -1),
    int3(0, 0, 1)
};

static const int3 CSGI_INDIRECT_DIRS[CSGI_TOTAL_DIRECTION_COUNT] = {
    int3(-1, 0, 0),
    int3(1, 0, 0),
    int3(0, -1, 0),
    int3(0, 1, 0),
    int3(0, 0, -1),
    int3(0, 0, 1),

    int3(-1, -1, -1),
    int3(1, -1, -1),
    int3(-1, 1, -1),
    int3(1, 1, -1),
    int3(-1, -1, 1),
    int3(1, -1, 1),
    int3(-1, 1, 1),
    int3(1, 1, 1),
};

static const float4 CSGI_CARDINAL_SUBRAY_TANGENT_WEIGHTS[CSGI_CARDINAL_SUBRAY_COUNT] = {
    float4(1.0, 1.0, 1.0, 1.0),
    float4(2.5, 0.15, 0.75, 0.75),
    float4(0.15, 2.5, 0.75, 0.75),
    float4(0.75, 0.75, 2.5, 0.15),
    float4(0.75, 0.75, 0.15, 2.5),
};

static const float3 CSGI_DIAGONAL_SUBRAY_TANGENT_WEIGHTS[CSGI_DIAGONAL_SUBRAY_COUNT] = {
    float3(0.5, 1.0, 1.0),
    float3(1.0, 0.5, 1.0),
    float3(1.0, 1.0, 0.5),
};

int3 csgi_dispatch_vx_to_local_vx(int3 dti, uint cascade_idx) {
    return
        dti < frame_constants.gi_cascades[cascade_idx].scroll_frac.xyz
        ? dti + CSGI_VOLUME_DIMS
        : dti;
}

int3 csgi_dispatch_vx_to_global_vx(int3 dti, uint cascade_idx) {
    int3 vx = csgi_dispatch_vx_to_local_vx(dti, cascade_idx);
    vx += frame_constants.gi_cascades[cascade_idx].scroll_int.xyz;
    return vx;
}

bool gi_volume_contains_vx(GiCascadeConstants vol, int3 vx) {
    const int3 volume_min = vol.scroll_int.xyz + vol.scroll_frac.xyz;
    const int3 volume_max = volume_min + CSGI_VOLUME_DIMS;
    return all(vx >= volume_min && vx < volume_max);
}

int3 gi_volume_get_cascade_outlier_offset(GiCascadeConstants vol, int3 vx) {
    const int3 volume_min = vol.scroll_int.xyz + vol.scroll_frac.xyz;
    const int3 volume_max = volume_min + CSGI_VOLUME_DIMS;
    return
        vx < volume_min
        ? -1
        : vx >= volume_max
        ? 1
        : 0;
}

float3 csgi_volume_center(uint cascade_idx) {
    const int3 volume_min_vx =
        frame_constants.gi_cascades[cascade_idx].scroll_int.xyz
        + frame_constants.gi_cascades[cascade_idx].scroll_frac.xyz;

    return
        (float3(volume_min_vx) + float(CSGI_VOLUME_DIMS) / 2.0)
        * csgi_voxel_size(cascade_idx)
        + CSGI_VOLUME_ORIGIN;
}

uint csgi_cascade_idx_for_pos(float3 pos) {
    // HACK: Don't sample the very edges as they can be leaky
    // TODO: find all the leaks :shrug:
    const float cascade_coverage_frac = 1.0 - 1.0 / CSGI_VOLUME_DIMS;

    for (uint i = 0; i < CSGI_CASCADE_COUNT - 1; ++i) {
        if (all(abs(pos - csgi_volume_center(i)) < 0.5 * cascade_coverage_frac * csgi_volume_size(i))) {
            return i;
        }
    }

    return CSGI_CASCADE_COUNT - 1;
}

float csgi_blended_cascade_idx_for_pos(float3 world_pos) {
    const float cascade_coverage_frac = 1.0 - 1.0 / CSGI_VOLUME_DIMS;

    for (uint i = 0; i < CSGI_CASCADE_COUNT - 1; ++i) {
        float3 pos = world_pos - csgi_volume_center(i);
        float3 extent = csgi_volume_size(i) * 0.5;

        float3 center_dist3 = abs(pos) / extent;
        float center_dist = max(center_dist3.x, max(center_dist3.y, center_dist3.z));
        float blend_region = 1.0 / CSGI_VOLUME_DIMS;
        float blend = smoothstep(1.0 - blend_region, 1.0, center_dist / cascade_coverage_frac);
        
        if (blend < 1.0) {
            return float(i) + blend;
        }
    }

    return CSGI_CASCADE_COUNT - 1;
}

float csgi_blended_voxel_size(float blended_cascade_idx) {
    float t = frac(blended_cascade_idx);
    float a = frame_constants.gi_cascades[floor(blended_cascade_idx)].voxel_size;
    float b = frame_constants.gi_cascades[floor(blended_cascade_idx) + 1].voxel_size;
    return t == 0.0 ? a : lerp(a, b, t);
}

uint3 csgi_wrap_vx_within_cascade(int3 vx) {
    // More general, works with non-power of two,
    // and in practice, there's no performance difference (at least on NV).
    // Should this change, can statically check CSGI_VOLUME_DIMS being power of two first.
    return (vx + CSGI_VOLUME_DIMS * 1024) % CSGI_VOLUME_DIMS;

    // Power of two grid sizes only
    //return vx & (CSGI_VOLUME_DIMS - 1);
}

bool csgi_was_dispatch_vx_just_scrolled_in(uint3 dispatch_vx, uint cascade_idx) {
    const int3 scroll_vx =
        csgi_dispatch_vx_to_local_vx(dispatch_vx, cascade_idx) - frame_constants.gi_cascades[cascade_idx].scroll_frac.xyz;
    const int3 scroll_offset = frame_constants.gi_cascades[cascade_idx].voxels_scrolled_this_frame.xyz;
    const int3 was_just_scrolled_in =
        scroll_offset > 0
        ? (scroll_vx + scroll_offset >= CSGI_VOLUME_DIMS)
        : (scroll_vx < -scroll_offset);

    return any(was_just_scrolled_in);
}

uint3 csgi_cardinal_vx_dir_subray_to_subray_vx(uint3 vx, uint dir_idx, uint subray)  {
    const int3 indirect_offset = int3(CSGI_CARDINAL_SUBRAY_COUNT * CSGI_VOLUME_DIMS * dir_idx, 0, 0);

#if CSGI_SUBRAY_PACKING_AS_SUBVOLUMES
    const int3 offset = int3(subray * CSGI_VOLUME_DIMS, 0, 0);
    const int3 stride = 1;
#else
    const int3 offset = int3(subray, 0, 0);
    const int3 stride = int3(CSGI_CARDINAL_SUBRAY_COUNT, 1, 1);
#endif

    return indirect_offset + offset + vx * stride;
}

uint3 csgi_diagonal_vx_dir_subray_to_subray_vx(uint3 vx, uint dir_idx, uint subray)  {
    const int3 indirect_offset = int3(
        CSGI_DIAGONAL_SUBRAY_COUNT * CSGI_VOLUME_DIMS * (dir_idx - CSGI_CARDINAL_DIRECTION_COUNT)
        + CSGI_DIAGONAL_DIRECTION_SUBRAY_OFFSET,
        0,
        0);

#if CSGI_SUBRAY_PACKING_AS_SUBVOLUMES
    const int3 offset = int3(subray * CSGI_VOLUME_DIMS, 0, 0);
    const int3 stride = 1;
#else
    const int3 offset = int3(subray, 0, 0);
    const int3 stride = int3(CSGI_DIAGONAL_SUBRAY_COUNT, 1, 1);
#endif

    return indirect_offset + offset + vx * stride;
}