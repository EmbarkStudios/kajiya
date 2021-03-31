#ifndef IMAGE_HLSL
#define IMAGE_HLSL

#include "curve.hlsl"

struct TextureImage {
    Texture2D<float4> _texture;
    int2 _size;

    static TextureImage from_parts(Texture2D<float4> texture, int2 size) {
        TextureImage res;
        res._texture = texture;
        res._size = size;
        return res;
    }

    float4 fetch(int2 px) {
        return _texture[px];
    }

    int2 size() {
        return _size;
    }
};

struct IdentityImageRemap {
    static IdentityImageRemap create() {
        IdentityImageRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return v;
    }
};

/// trait Remap {
///     float4 remap(float4 v);
/// }
template<typename Remap>
float3 image_sample_catmull_rom(TextureImage img, float2 P, Remap remap = IdentityImageRemap::create()) {
    // https://www.shadertoy.com/view/MllSzX

    float2 pixel = P * img.size() + 0.5;
    float2 c_onePixel = 1.0 / img.size();
    float2 c_twoPixels = c_onePixel * 2.0;
    
    float2 frc = frac(pixel);
    //pixel = floor(pixel) / output_tex_size.xy - float2(c_onePixel/2.0);
    int2 ipixel = int2(pixel) - 1;
    
    float3 C00 = remap.remap(img.fetch(ipixel + int2(-1 ,-1))).rgb;
    float3 C10 = remap.remap(img.fetch(ipixel + int2( 0, -1))).rgb;
    float3 C20 = remap.remap(img.fetch(ipixel + int2( 1, -1))).rgb;
    float3 C30 = remap.remap(img.fetch(ipixel + int2( 2,-1))).rgb;
    
    float3 C01 = remap.remap(img.fetch(ipixel + int2(-1 , 0))).rgb;
    float3 C11 = remap.remap(img.fetch(ipixel + int2( 0, 0))).rgb;
    float3 C21 = remap.remap(img.fetch(ipixel + int2( 1, 0))).rgb;
    float3 C31 = remap.remap(img.fetch(ipixel + int2( 2, 0))).rgb;
    
    float3 C02 = remap.remap(img.fetch(ipixel + int2(-1 , 1))).rgb;
    float3 C12 = remap.remap(img.fetch(ipixel + int2( 0, 1))).rgb;
    float3 C22 = remap.remap(img.fetch(ipixel + int2( 1, 1))).rgb;
    float3 C32 = remap.remap(img.fetch(ipixel + int2( 2, 1))).rgb;
    
    float3 C03 = remap.remap(img.fetch(ipixel + int2(-1 , 2))).rgb;
    float3 C13 = remap.remap(img.fetch(ipixel + int2( 0, 2))).rgb;
    float3 C23 = remap.remap(img.fetch(ipixel + int2( 1 , 2))).rgb;
    float3 C33 = remap.remap(img.fetch(ipixel + int2( 2, 2))).rgb;
    
    float3 CP0X = cubic_hermite(C00, C10, C20, C30, frc.x);
    float3 CP1X = cubic_hermite(C01, C11, C21, C31, frc.x);
    float3 CP2X = cubic_hermite(C02, C12, C22, C32, frc.x);
    float3 CP3X = cubic_hermite(C03, C13, C23, C33, frc.x);
    
    return cubic_hermite(CP0X, CP1X, CP2X, CP3X, frc.y);
}


#endif // IMAGE_HLSL
