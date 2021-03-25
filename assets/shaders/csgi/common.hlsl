#define USE_RTDGI_CONTROL_VARIATES 1

static const uint CSGI_SLICE_COUNT = 6;
static const uint CSGI_INDIRECT_COUNT = 6 + 8;

static const uint CSGI_VOLUME_DIMS = 64;

#define CSGI_SCENE_CORNELL_BOX 0
#define CSGI_SCENE_BATTLE 1
#define CSGI_SCENE_GAS_STATIONS 2
#define CSGI_SCENE_VIZIERS 3
#define CSGI_SCENE_SPONZA 4
#define CSGI_SCENE_MISC 5

#define CSGI_SCENE CSGI_SCENE_MISC

#if CSGI_SCENE == CSGI_SCENE_BATTLE
    #define CSGI_VOLUME_CENTER float3(-15, 4.0, 1 /*8*/)
    static const float CSGI_VOLUME_SIZE = 16.0;
#elif CSGI_SCENE == CSGI_SCENE_CORNELL_BOX
    #define CSGI_VOLUME_CENTER float3(0, 1.0, 0)
    //static const float CSGI_VOLUME_SIZE = CSGI_VOLUME_DIMS * 0.04;
    static const float CSGI_VOLUME_SIZE = CSGI_VOLUME_DIMS * 0.1;
#elif CSGI_SCENE == CSGI_SCENE_GAS_STATIONS
    //#define CSGI_VOLUME_CENTER float3(2, 8, 6)
    #define CSGI_VOLUME_CENTER float3(9, 2, 2)
    static const float CSGI_VOLUME_SIZE = 24.0;
#elif CSGI_SCENE == CSGI_SCENE_VIZIERS
    #define CSGI_VOLUME_CENTER float3(0, 15, 0)
    static const float CSGI_VOLUME_SIZE = CSGI_VOLUME_DIMS * (1.0 / 4);
#elif CSGI_SCENE == CSGI_SCENE_SPONZA
    #define CSGI_VOLUME_CENTER float3(0, 0, 0)
    static const float CSGI_VOLUME_SIZE = CSGI_VOLUME_DIMS * 0.16;
#else
    #define CSGI_VOLUME_CENTER float3(0, 0, 0)
    static const float CSGI_VOLUME_SIZE = 12;
#endif

//static const float CSGI_VOLUME_SCALE = CSGI_VOLUME_SIZE / CSGI_VOLUME_DIMS;

#define CSGI_VOXEL_SIZE (float(CSGI_VOLUME_SIZE) / float(CSGI_VOLUME_DIMS))

/*float3 gi_volume_center(float3x3 slice_rot) {
    #ifdef CSGI_VOLUME_CENTER
        return CSGI_VOLUME_CENTER;
    #else
        float3 pos = get_eye_position();
        pos = mul(pos, slice_rot);
        pos = trunc(pos / CSGI_VOXEL_SIZE) * CSGI_VOXEL_SIZE;
        pos = mul(slice_rot, pos);
        return pos;
    #endif
}*/

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
