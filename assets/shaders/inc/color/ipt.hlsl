#ifndef NOTORIOUS6_IPT_HLSL
#define NOTORIOUS6_IPT_HLSL

float3 XYZ_to_IPT(float3 xyz) {
    float3 lms = mul(
        float3x3(
             0.4002, 0.7075, -0.0807,
            -0.2280, 1.1500,  0.0612,
             0.0,       0.0,  0.9184),
        xyz
    );

    lms.x = select(lms.x >= 0.0, pow(lms.x, 0.43), -pow(-lms.x, 0.43));
    lms.y = select(lms.y >= 0.0, pow(lms.y, 0.43), -pow(-lms.y, 0.43));
    lms.z = select(lms.z >= 0.0, pow(lms.z, 0.43), -pow(-lms.z, 0.43));

    return mul(
        float3x3(
         0.4000,  0.4000, 0.2000,
         4.4550, -4.8510, 0.3960,
         0.8056, 0.3572, -1.1628),
        lms
    );
}


float3 IPT_to_XYZ(float3 ipt) {
    float3 lms = mul(
        float3x3(
         1.0,  0.0976, 0.2052,
         1.0, -0.1139, 0.1332,
         1.0, 0.0326, -0.6769),
        ipt
    );

    lms.x = select(lms.x >= 0.0, pow(lms.x, 1.0 / 0.43), -pow(-lms.x, 1.0 / 0.43));
    lms.y = select(lms.y >= 0.0, pow(lms.y, 1.0 / 0.43), -pow(-lms.y, 1.0 / 0.43));
    lms.z = select(lms.z >= 0.0, pow(lms.z, 1.0 / 0.43), -pow(-lms.z, 1.0 / 0.43));

    return mul(
        float3x3(
         1.8501, -1.1383, 0.2385,
         0.3668, 0.6439,  -0.0107,
         0.0,       0.0,  1.0889),
        lms
    );
}

#endif  // NOTORIOUS6_IPT_HLSL
