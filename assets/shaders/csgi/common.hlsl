#define GI_SCENE_CORNELL_BOX 0
#define GI_SCENE_BATTLE 1
#define GI_SCENE_GAS_STATIONS 2

#define GI_SCENE GI_SCENE_GAS_STATIONS

#if GI_SCENE == GI_SCENE_BATTLE
    static const float3 GI_VOLUME_CENTER = float3(-15, 4.0, 8);
    static const uint GI_VOLUME_DIMS = 32;
    static const float GI_VOLUME_SCALE = 1.0 / 4;
#elif GI_SCENE == GI_SCENE_CORNELL_BOX
    static const float3 GI_VOLUME_CENTER = float3(0, 1.0, 0);
    static const uint GI_VOLUME_DIMS = 32;
    static const float GI_VOLUME_SCALE = 1.0 / 8;
#elif GI_SCENE == GI_SCENE_GAS_STATIONS
    //static const float3 GI_VOLUME_CENTER = float3(2, 8, 6);
    static const float3 GI_VOLUME_CENTER = float3(9, 2, 2);
    static const uint GI_VOLUME_DIMS = 32;
    static const float GI_VOLUME_SCALE = 1.0 / 2;
#endif


static const uint DEBUG_SLICE_IDX = 0;
static const uint GI_SLICE_COUNT = 16;