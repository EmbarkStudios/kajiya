#define USE_RTDGI_CONTROL_VARIATES 1

static const uint CSGI_SLICE_COUNT = 6;
static const uint CSGI_INDIRECT_COUNT = 6 + 8;

static const uint CSGI_VOLUME_DIMS = 64;

#define CSGI_VOLUME_CENTER float3(0, 0, 0)
static const float CSGI_VOLUME_SIZE = 12;
#define CSGI_VOXEL_SIZE (float(CSGI_VOLUME_SIZE) / float(CSGI_VOLUME_DIMS))

static const uint CSGI_NEIGHBOR_DIR_COUNT = 9;
static const int3 CSGI_NEIGHBOR_DIRS[CSGI_NEIGHBOR_DIR_COUNT] = {
    int3(0, 0, -1),

    int3(1, 0, -1),
    int3(-1, 0, -1),
    int3(0, 1, -1),
    int3(0, -1, -1),

    int3(1, 1, -1),
    int3(-1, 1, -1),
    int3(1, -1, -1),
    int3(-1, -1, -1)
};

static const int3 CSGI_SLICE_DIRS[CSGI_SLICE_COUNT] = {
    int3(-1, 0, 0),
    int3(1, 0, 0),
    int3(0, -1, 0),
    int3(0, 1, 0),
    int3(0, 0, -1),
    int3(0, 0, 1)
};

static const int3 CSGI_INDIRECT_DIRS[CSGI_INDIRECT_COUNT] = {
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
