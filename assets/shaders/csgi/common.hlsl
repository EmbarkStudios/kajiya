#if 1
    static const float3x3 slice_rot = float3x3(
        0, 0, 1,
        0, 1, 0,
        1, 0, 0
    );
#else
    static const float3x3 slice_rot = float3x3(
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    );
#endif

static const float3 GI_VOLUME_CENTER = float3(0, 1, 0);
static const uint GI_VOLUME_DIMS = 32;
static const float GI_VOLUME_SCALE = 1.0 / 16.0;