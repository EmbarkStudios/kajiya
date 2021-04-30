#include "../inc/frame_constants.hlsl"

#define USE_RTDGI_CONTROL_VARIATES 1

static const uint CSGI_VOLUME_DIMS = 64;

static const uint CSGI_CARDINAL_DIRECTION_COUNT = 6;
static const uint CSGI_CARDINAL_SUBRAY_COUNT = 5;

static const uint CSGI_DIAGONAL_DIRECTION_COUNT = 8;
static const uint CSGI_DIAGONAL_SUBRAY_COUNT = 3;

// X coord in the image of the first subray for diagonal directions.
// Basically after all the cardinal subrays.
static const uint CSGI_DIAGONAL_DIRECTION_SUBRAY_OFFSET =
    CSGI_VOLUME_DIMS * CSGI_CARDINAL_DIRECTION_COUNT * CSGI_CARDINAL_SUBRAY_COUNT;

static const uint CSGI_TOTAL_DIRECTION_COUNT = CSGI_CARDINAL_DIRECTION_COUNT + CSGI_DIAGONAL_DIRECTION_COUNT;

//#define CSGI_ACCUM_HYSTERESIS 0.05
//#define CSGI_ACCUM_HYSTERESIS 0.15
#define CSGI_ACCUM_HYSTERESIS 0.25
//#define CSGI_ACCUM_HYSTERESIS 1.0

// Note: the "csgi subray combine" pass must be enabled if this is 0.
#define CSGI_SUBRAY_COMBINE_DURING_SWEEP 1

#define CSGI_VOLUME_CENTER float3(0, 0, 0)
//static const float CSGI_VOLUME_SIZE = 12;
#define CSGI_VOLUME_SIZE (12.0 * frame_constants.world_gi_scale)
#define CSGI_VOXEL_SIZE (float(CSGI_VOLUME_SIZE) / float(CSGI_VOLUME_DIMS))

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
