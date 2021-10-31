#ifndef RAY_CONE_HLSL
#define RAY_CONE_HLSL

// https://media.contentapi.ea.com/content/dam/ea/seed/presentations/2019-ray-tracing-gems-chapter-20-akenine-moller-et-al.pdf
struct RayCone {
    float width;
    float spread_angle;

    static RayCone from_spread_angle(float spread_angle) {
        RayCone res;
        res.width = 0.0;
        res.spread_angle = spread_angle;
        return res;
    }

    static RayCone from_width_spread_angle(float width, float spread_angle) {
        RayCone res;
        res.width = width;
        res.spread_angle = spread_angle;
        return res;
    }

    RayCone propagate(float surface_spread_angle, float hit_t) {
        RayCone res;
        res.width = this.spread_angle * hit_t + this.width;
        res.spread_angle = this.spread_angle + surface_spread_angle;
        return res;
    }

    float width_at_t(float hit_t) {
        return this.width + this.spread_angle * hit_t;
    }
};

#endif  // RAY_CONE_HLSL