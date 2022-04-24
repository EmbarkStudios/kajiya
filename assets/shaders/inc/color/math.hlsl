#ifndef NOTORIOUS6_MATH_HLSL
#define NOTORIOUS6_MATH_HLSL

#define M_PI 3.1415926535897932384626433832795

float cbrt(float x) {
    return sign(x) * pow(abs(x),1.0 / 3.0);
}

float soft_shoulder(float x, float max_val, float p) {
    return x / pow(pow(x / max_val, p) + 1.0, 1.0 / p);
}

float catmull_rom(float x, float v0,float v1, float v2,float v3) {
	float c2 = -.5 * v0	+ 0.5*v2;
	float c3 = v0		+ -2.5*v1 + 2.0*v2 + -.5*v3;
	float c4 = -.5 * v0	+ 1.5*v1 + -1.5*v2 + 0.5*v3;
	return(((c4 * x + c3) * x + c2) * x + v1);
}

#endif  // NOTORIOUS6_MATH_HLSL
