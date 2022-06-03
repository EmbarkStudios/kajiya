#include "../rtdgi/near_field_settings.hlsl"

#define USE_AO_ONLY 1
static const uint SSGI_HALF_SAMPLE_COUNT = 3;
#define SSGI_KERNEL_RADIUS (SSGI_NEAR_FIELD_RADIUS * output_tex_size.w)
#define MAX_KERNEL_RADIUS_CS 10
#define USE_KERNEL_DISTANCE_SCALING 0
#define USE_RANDOM_JITTER 0
#define USE_SSGI_FACING_CORRECTION 1

#define SSGI_FULLRES

#include "ssgi.hlsl"