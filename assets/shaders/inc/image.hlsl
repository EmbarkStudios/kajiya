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
float4 image_sample_catmull_rom(TextureImage img, float2 P, Remap remap = IdentityImageRemap::create()) {
    // https://www.shadertoy.com/view/MllSzX

    float2 pixel = P * img.size() + 0.5;
    float2 c_onePixel = 1.0 / img.size();
    float2 c_twoPixels = c_onePixel * 2.0;
    
    float2 frc = frac(pixel);
    //pixel = floor(pixel) / output_tex_size.xy - float2(c_onePixel/2.0);
    int2 ipixel = int2(pixel) - 1;
    
    float4 C00 = remap.remap(img.fetch(ipixel + int2(-1 ,-1)));
    float4 C10 = remap.remap(img.fetch(ipixel + int2( 0, -1)));
    float4 C20 = remap.remap(img.fetch(ipixel + int2( 1, -1)));
    float4 C30 = remap.remap(img.fetch(ipixel + int2( 2,-1)));
    
    float4 C01 = remap.remap(img.fetch(ipixel + int2(-1 , 0)));
    float4 C11 = remap.remap(img.fetch(ipixel + int2( 0, 0)));
    float4 C21 = remap.remap(img.fetch(ipixel + int2( 1, 0)));
    float4 C31 = remap.remap(img.fetch(ipixel + int2( 2, 0)));
    
    float4 C02 = remap.remap(img.fetch(ipixel + int2(-1 , 1)));
    float4 C12 = remap.remap(img.fetch(ipixel + int2( 0, 1)));
    float4 C22 = remap.remap(img.fetch(ipixel + int2( 1, 1)));
    float4 C32 = remap.remap(img.fetch(ipixel + int2( 2, 1)));
    
    float4 C03 = remap.remap(img.fetch(ipixel + int2(-1 , 2)));
    float4 C13 = remap.remap(img.fetch(ipixel + int2( 0, 2)));
    float4 C23 = remap.remap(img.fetch(ipixel + int2( 1 , 2)));
    float4 C33 = remap.remap(img.fetch(ipixel + int2( 2, 2)));
    
    float4 CP0X = cubic_hermite(C00, C10, C20, C30, frc.x);
    float4 CP1X = cubic_hermite(C01, C11, C21, C31, frc.x);
    float4 CP2X = cubic_hermite(C02, C12, C22, C32, frc.x);
    float4 CP3X = cubic_hermite(C03, C13, C23, C33, frc.x);
    
    return cubic_hermite(CP0X, CP1X, CP2X, CP3X, frc.y);
}

/// trait Remap {
///     float4 remap(float4 v);
/// }
template<typename Remap>
float4 image_sample_catmull_rom_approx(in Texture2D<float4> tex, in SamplerState linearSampler, in float2 uv, in float2 texSize, bool useCornerTaps, Remap remap = IdentityImageRemap::create()) {
    // https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1

    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    float2 samplePos = uv * texSize;
    float2 texPos1 = floor(samplePos - 0.5f) + 0.5f;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    float2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    float2 w12 = w1 + w2;
    float2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    float2 texPos0 = texPos1 - 1;
    float2 texPos3 = texPos1 + 2;
    float2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    float4 result = 0.0f;

    if (useCornerTaps) {
        result += remap.remap(tex.SampleLevel(linearSampler, float2(texPos0.x, texPos0.y), 0.0f)) * w0.x * w0.y;
    }

    result += remap.remap(tex.SampleLevel(linearSampler, float2(texPos12.x, texPos0.y), 0.0f)) * w12.x * w0.y;

    if (useCornerTaps) {
        result += remap.remap(tex.SampleLevel(linearSampler, float2(texPos3.x, texPos0.y), 0.0f)) * w3.x * w0.y;
    }

    result += remap.remap(tex.SampleLevel(linearSampler, float2(texPos0.x, texPos12.y), 0.0f)) * w0.x * w12.y;
    result += remap.remap(tex.SampleLevel(linearSampler, float2(texPos12.x, texPos12.y), 0.0f)) * w12.x * w12.y;
    result += remap.remap(tex.SampleLevel(linearSampler, float2(texPos3.x, texPos12.y), 0.0f)) * w3.x * w12.y;

    if (useCornerTaps) {
        result += remap.remap(tex.SampleLevel(linearSampler, float2(texPos0.x, texPos3.y), 0.0f)) * w0.x * w3.y;
    }

    result += remap.remap(tex.SampleLevel(linearSampler, float2(texPos12.x, texPos3.y), 0.0f)) * w12.x * w3.y;

    if (useCornerTaps) {
        result += remap.remap(tex.SampleLevel(linearSampler, float2(texPos3.x, texPos3.y), 0.0f)) * w3.x * w3.y;
    }

    if (!useCornerTaps) {
        result /= (w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y);
    }

    return result;
}

/// trait Remap {
///     float4 remap(float4 v);
/// }
template<typename Remap>
float4 image_sample_catmull_rom_9tap(in Texture2D<float4> tex, in SamplerState linearSampler, in float2 uv, in float2 texSize, Remap remap = IdentityImageRemap::create()) {
    return image_sample_catmull_rom_approx(
        tex, linearSampler, uv, texSize, true, remap
    );
}

/// trait Remap {
///     float4 remap(float4 v);
/// }
template<typename Remap>
float4 image_sample_catmull_rom_5tap(in Texture2D<float4> tex, in SamplerState linearSampler, in float2 uv, in float2 texSize, Remap remap = IdentityImageRemap::create()) {
    return image_sample_catmull_rom_approx(
        tex, linearSampler, uv, texSize, false, remap
    );
}

#endif // IMAGE_HLSL
