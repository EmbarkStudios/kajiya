#ifndef NOTORIOUS6_ICTCP_HLSL
#define NOTORIOUS6_ICTCP_HLSL

// From https://www.shadertoy.com/view/ldKcz3

static const float PQ_C1 = 0.8359375f;			// 3424.f / 4096.f;
static const float PQ_C2 = 18.8515625f;		// 2413.f / 4096.f * 32.f;
static const float PQ_C3 = 18.6875f;			// 2392.f / 4096.f * 32.f;
static const float PQ_M1 = 0.159301758125f;	// 2610.f / 4096.f / 4;
static const float PQ_M2 = 78.84375f;			// 2523.f / 4096.f * 128.f;
static const float PQ_MAX = 10000.0;

// PQ_OETF - Optical-Electro Transfer Function 

float linear_to_PQ( float linearValue )
{
	float L = linearValue / PQ_MAX;
	float Lm1 = pow( L, PQ_M1 );
	float X = ( PQ_C1 + PQ_C2 * Lm1 ) / ( 1.0f + PQ_C3 * Lm1 );
	float pqValue = pow( X, PQ_M2 );
	return pqValue;
}

float3 linear_to_PQ( float3 linearValues )
{
	float3 L = linearValues / PQ_MAX;
	float3 Lm1 = pow( max(0.0.xxx, L.xyz), PQ_M1.xxx );
	float3 X = ( PQ_C1 + PQ_C2 * Lm1 ) / ( 1.0f + PQ_C3 * Lm1 );
	float3 pqValues = pow( max(0.0.xxx, X), PQ_M2.xxx );
	return pqValues;
}

// PQ_EOTF - Electro-Optical Transfer Function 

float PQ_to_linear( float pqValue )
{
	float M = PQ_C2 - PQ_C3 * pow( max(0.0, pqValue), 1.0 / PQ_M2 );
	float N = max( pow( max(0.0, pqValue), 1.0f / PQ_M2 ) - PQ_C1, 0.0f );
	float L = pow( N / M, 1.0f / PQ_M1 );
	float linearValue = L * PQ_MAX;
	return linearValue;
}

float3 PQ_to_linear( float3 pqValues )
{
	float3 M = PQ_C2 - PQ_C3 * pow( max(0.0.xxx, pqValues), 1. / PQ_M2.xxx );
	float3 N = max( pow( max(0.0.xxx, pqValues), 1. / PQ_M2.xxx ) - PQ_C1, 0.0f );
	float3 L = pow( max(0.0.xxx, N / M), 1. / PQ_M1.xxx );
	float3 linearValues = L * PQ_MAX;
	return linearValues;
}


// BT.709 <-> BT.2020 Primaries

float3 BT709_to_BT2020( float3 linearBT709 )
{
	float3 linearBT2020 = mul(
        float3x3(
            0.6274,    0.3293,    0.0433,
            0.0691,    0.9195,    0.0114,
            0.0164,    0.0880,    0.8956
        ),
		linearBT709.rgb);
	return linearBT2020;
}

float3 BT2020_to_BT709( float3 linearBT2020 )
{
	float3 linearBT709 = mul( 
    	float3x3(
             1.6605,	-0.5877,	-0.0728,
            -0.1246,	 1.1330,	-0.0084,
            -0.0182,	-0.1006,	 1.1187
        ),
		linearBT2020.rgb);
	return linearBT709;
}

// LMS <-> BT2020

float3 BT2020_to_LMS( float3 linearBT2020 )
{
	float R = linearBT2020.r;
	float G = linearBT2020.g;
	float B = linearBT2020.b;

    float L = 0.4121093750000000f * R + 0.5239257812500000f * G + 0.0639648437500000f * B;
    float M = 0.1667480468750000f * R + 0.7204589843750000f * G + 0.1127929687500000f * B;
    float S = 0.0241699218750000f * R + 0.0754394531250000f * G + 0.9003906250000000f * B;

	float3 linearLMS = float3(L, M, S);
	return linearLMS;
}

float3 LMS_to_BT2020( float3 linearLMS )
{
	float L = linearLMS.x;
	float M = linearLMS.y;
	float S = linearLMS.z;

	float R =  3.4366066943330793f * L - 2.5064521186562705f * M + 0.0698454243231915f * S;
	float G = -0.7913295555989289f * L + 1.9836004517922909f * M - 0.1922708961933620f * S;
    float B = -0.0259498996905927f * L - 0.0989137147117265f * M + 1.1248636144023192f * S;

	float3 linearBT2020 = float3(R, G, B);
	return linearBT2020;
}

// Misc. Color Space Conversion

// ICtCp <-> PQ LMS

float3 PQ_LMS_to_ICtCp( float3 PQ_LMS )
{
	float L = PQ_LMS.x;
	float M = PQ_LMS.y;
	float S = PQ_LMS.z;

    float I  = 0.5f * L + 0.5f * M;
    float Ct = 1.613769531250000f * L - 3.323486328125000f * M + 1.709716796875000f * S;
    float Cp = 4.378173828125000f * L - 4.245605468750000f * M - 0.132568359375000f * S;

	float3 ICtCp = float3(I, Ct, Cp);
	return ICtCp;
}

float3 ICtCp_to_PQ_LMS( float3 ICtCp )
{
	float I  = ICtCp.x;
	float Ct = ICtCp.y;
	float Cp = ICtCp.z;

	float L = I + 0.00860903703793281f * Ct + 0.11102962500302593f * Cp;
	float M = I - 0.00860903703793281f * Ct - 0.11102962500302593f * Cp;
	float S = I + 0.56003133571067909f * Ct - 0.32062717498731880f * Cp;

	float3 PQ_LMS = float3(L, M, S);
	return PQ_LMS;
}

// Linear BT2020 <-> ICtCp
// 
// https://www.dolby.com/us/en/technologies/dolby-vision/ictcp-white-paper.pdf
// http://www.jonolick.com/home/hdr-videos-part-2-colors

float3 BT2020_to_ICtCp( float3 linearBT2020 ) 
{
	float3 LMS = BT2020_to_LMS( linearBT2020 );
	float3 PQ_LMS = linear_to_PQ( LMS );
	float3 ICtCp = PQ_LMS_to_ICtCp( PQ_LMS );

	return ICtCp;
}

float3 ICtCp_to_BT2020( float3 ICtCp )
{
	float3 PQ_LMS = ICtCp_to_PQ_LMS( ICtCp );
	float3 LMS = PQ_to_linear( PQ_LMS );
	float3 linearBT2020 = LMS_to_BT2020( LMS );
	return linearBT2020;
}

// ----------------------------------------------------------------

float3 BT709_to_ICtCp( float3 linearBT709 ) 
{
	float3 linearBT2020 = BT709_to_BT2020(linearBT709);
	float3 LMS = BT2020_to_LMS( linearBT2020 );
	float3 PQ_LMS = linear_to_PQ( LMS );
	float3 ICtCp = PQ_LMS_to_ICtCp( PQ_LMS );

	return ICtCp;
}

float3 ICtCp_to_BT709( float3 ICtCp )
{
	float3 PQ_LMS = ICtCp_to_PQ_LMS( ICtCp );
	float3 LMS = PQ_to_linear( PQ_LMS );
	float3 linearBT2020 = LMS_to_BT2020( LMS );
	float3 linearBT709 = BT2020_to_BT709( linearBT2020 );
	return linearBT709;
}

#endif  // NOTORIOUS6_ICTCP_HLSL
