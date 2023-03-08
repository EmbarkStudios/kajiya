// from Alex Tardiff: http://alextardif.com/Lightness.html

#ifndef NOTORIOUS6_LAB_HLSL
#define NOTORIOUS6_LAB_HLSL

#include "xyz.hlsl"

float degrees_to_radians(float degrees) 
{
    return degrees * M_PI / 180.0;
}

float radians_to_degrees(float radians) 
{
    return radians * (180.0 / M_PI);
}

// http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
float3 XYZ_to_LAB(float3 xyz) 
{
    const float3 D65_XYZ = float3(0.9504, 1.0000, 1.0888);
    
    xyz /= D65_XYZ;
    xyz = float3(
        select(xyz.x > 0.008856, pow(abs(xyz.x), 1.0 / 3.0), xyz.x * 7.787 + 16.0 / 116.0),
        select(xyz.y > 0.008856, pow(abs(xyz.y), 1.0 / 3.0), xyz.y * 7.787 + 16.0 / 116.0),
        select(xyz.z > 0.008856, pow(abs(xyz.z), 1.0 / 3.0), xyz.z * 7.787 + 16.0 / 116.0)
    );

    float l = 116.0 * xyz.y - 16.0;
    float a = 500.0 * (xyz.x - xyz.y);
    float b = 200.0 * (xyz.y - xyz.z);
    
    return float3(l, a, b);
}

// http://www.brucelindbloom.com/index.html?Eqn_Lab_to_XYZ.html
float3 LABToXYZ(float3 lab)
{
	const float3 D65_XYZ = float3(0.9504, 1.0000, 1.0888);

    float fy = (lab[0] + 16.0) / 116.0;
    float fx = lab[1] / 500.0 + fy;
    float fz = fy - lab[2] / 200.0;
	float3 fxyz = float3(fx, fy, fz);

	float3 xyz = fxyz * fxyz * fxyz;
	return D65_XYZ * float3(
        select(fxyz.x > 0.206893, xyz.x, (116.0 * fxyz.x - 16.0) / 903.3),
        select(fxyz.y > 0.206893, xyz.y, (116.0 * fxyz.y - 16.0) / 903.3),
        select(fxyz.z > 0.206893, xyz.z, (116.0 * fxyz.z - 16.0) / 903.3)
    );
}

//http://www.brucelindbloom.com/index.html?Eqn_Lab_to_LCH.html
float3 LAB_to_Lch(float3 lab) 
{
    float c = sqrt(lab.y * lab.y + lab.z * lab.z);
    float h = atan2(lab.z, lab.y);
    
    if (h >= 0.0) {
        h = radians_to_degrees(h);
    } else {
        h = radians_to_degrees(h) + 360.0;
    }
    
    return float3(lab.x, c, h);
}

#endif  // NOTORIOUS6_LAB_HLSL
