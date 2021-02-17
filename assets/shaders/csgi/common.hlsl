static const uint GI_SLICE_COUNT = 16;
static const uint GI_PRETRACE_COUNT = 32;

static const uint GI_VOLUME_DIMS = 32;
static const uint GI_PRETRACE_DIMS = 32;

#define GI_SCENE_CORNELL_BOX 0
#define GI_SCENE_BATTLE 1
#define GI_SCENE_GAS_STATIONS 2
#define GI_SCENE_VIZIERS 3
#define GI_SCENE_SPONZA 4
#define GI_SCENE_MISC 5

#define GI_SCENE GI_SCENE_BATTLE

#if GI_SCENE == GI_SCENE_BATTLE
    //#define GI_VOLUME_CENTER float3(-15, 4.0, 8)
    static const float GI_VOLUME_SIZE = 10.0;
#elif GI_SCENE == GI_SCENE_CORNELL_BOX
    #define GI_VOLUME_CENTER float3(0, 1.0, 0)
    static const float GI_VOLUME_SIZE = GI_VOLUME_DIMS * (1.0 / 8);
#elif GI_SCENE == GI_SCENE_GAS_STATIONS
    //#define GI_VOLUME_CENTER float3(2, 8, 6)
    #define GI_VOLUME_CENTER float3(9, 2, 2)
    static const float GI_VOLUME_SIZE = GI_VOLUME_DIMS * (1.0 / 2);
#elif GI_SCENE == GI_SCENE_VIZIERS
    #define GI_VOLUME_CENTER float3(0, 15, 0)
    static const float GI_VOLUME_SIZE = GI_VOLUME_DIMS * (1.0 / 3);
#elif GI_SCENE == GI_SCENE_SPONZA
    //#define GI_VOLUME_CENTER float3(0, 4, -4)
    static const float GI_VOLUME_SIZE = GI_VOLUME_DIMS * 0.75;
#else
    static const float GI_VOLUME_SIZE = 24.0;
#endif

//static const float GI_VOLUME_SCALE = GI_VOLUME_SIZE / GI_VOLUME_DIMS;

#define GI_VOXEL_SIZE (float(GI_VOLUME_SIZE) / float(GI_VOLUME_DIMS))

float3 gi_volume_center(float3x3 slice_rot) {
    #ifdef GI_VOLUME_CENTER
        return GI_VOLUME_CENTER;
    #else
        float3 pos = get_eye_position();
        pos = mul(pos, slice_rot);
        pos = trunc(pos / GI_VOXEL_SIZE) * GI_VOXEL_SIZE;
        pos = mul(slice_rot, pos);
        return pos;
    #endif
}

static const uint GI_NEIGHBOR_DIR_COUNT = 9;
static const int3 GI_NEIGHBOR_DIRS[GI_NEIGHBOR_DIR_COUNT] = {
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
