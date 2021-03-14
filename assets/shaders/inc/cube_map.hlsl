static const float3x3 CUBE_MAP_FACE_ROTATIONS[6] = {
    float3x3(0,0,-1, 0,-1,0, -1,0,0),   // right
    float3x3(0,0,1, 0,-1,0, 1,0,0),     // left

    float3x3(1,0,0, 0,0,-1, 0,1,0),     // top
    float3x3(1,0,0, 0,0,1, 0,-1,0),     // bottom

    float3x3(1,0,0, 0,-1,0, 0,0,-1),    // back
    float3x3(-1,0,0, 0,-1,0, 0,0,1),    // front
};