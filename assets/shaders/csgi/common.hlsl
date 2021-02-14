static const uint GI_SLICE_COUNT = 16;
static const uint GI_PRETRACE_COUNT = 32;

static const uint GI_VOLUME_DIMS = 32;
static const uint GI_PRETRACE_DIMS = 48;

#define GI_SCENE_CORNELL_BOX 0
#define GI_SCENE_BATTLE 1
#define GI_SCENE_GAS_STATIONS 2
#define GI_SCENE_VIZIERS 3
#define GI_SCENE_SPONZA 4

#define GI_SCENE GI_SCENE_BATTLE

#if GI_SCENE == GI_SCENE_BATTLE
    static const float3 GI_VOLUME_CENTER = float3(-15, 4.0, 8);
    static const float GI_VOLUME_SIZE = 10.0;
#elif GI_SCENE == GI_SCENE_CORNELL_BOX
    static const float3 GI_VOLUME_CENTER = float3(0, 1.0, 0);
    static const float GI_VOLUME_SIZE = GI_VOLUME_DIMS * (1.0 / 8);
#elif GI_SCENE == GI_SCENE_GAS_STATIONS
    //static const float3 GI_VOLUME_CENTER = float3(2, 8, 6);
    static const float3 GI_VOLUME_CENTER = float3(9, 2, 2);
    static const float GI_VOLUME_SIZE = GI_VOLUME_DIMS * (1.0 / 2);
#elif GI_SCENE == GI_SCENE_VIZIERS
    static const float3 GI_VOLUME_CENTER = float3(0, 15, 0);
    static const float GI_VOLUME_SIZE = GI_VOLUME_DIMS * (1.0 / 3);
#elif GI_SCENE == GI_SCENE_SPONZA
    //static const float3 GI_VOLUME_CENTER = float3(0, 4, -4);
    //static const float GI_VOLUME_SIZE = GI_VOLUME_DIMS * 0.333;

    static const float3 GI_VOLUME_CENTER = float3(0, 0, 0);
    static const float GI_VOLUME_SIZE = GI_VOLUME_DIMS * 0.75;
#endif

//static const float GI_VOLUME_SCALE = GI_VOLUME_SIZE / GI_VOLUME_DIMS;