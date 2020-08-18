#include "sdf_consts.inc"

uniform restrict writeonly image3D outputTex;

layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main() {
    ivec3 pix = ivec3(gl_GlobalInvocationID.xyz);
    imageStore(outputTex, pix, SDF_EMPTY_DIST.xxxx);
}
