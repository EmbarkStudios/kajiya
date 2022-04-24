#ifndef NOTORIOUS6_XYZ_HLSL
#define NOTORIOUS6_XYZ_HLSL

float3 CIE_xyY_xy_to_xyz(float2 xy) {
    return float3( xy, 1.0f - xy.x - xy.y );
}

float3 CIE_xyY_to_XYZ( float3 CIE_xyY )
{
    float x = CIE_xyY[0];
    float y = CIE_xyY[1];
    float Y = CIE_xyY[2];
    
    float X = (Y / y) * x;
    float Z = (Y / y) * (1.0 - x - y);
        
    return float3( X, Y, Z );        
}

float3 CIE_XYZ_to_xyY( float3 CIE_XYZ )
{
    float X = CIE_XYZ[0];
    float Y = CIE_XYZ[1];
    float Z = CIE_XYZ[2];
    
    float N = X + Y + Z;
    
    float x = X / N;
    float y = Y / N;
    float z = Z / N;
    
    return float3(x,y,Y);
}

static const float2 white_D65_xy = float2(0.31271, 0.32902);

#endif  // NOTORIOUS6_XYZ_HLSL
