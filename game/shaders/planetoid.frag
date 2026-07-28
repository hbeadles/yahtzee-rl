#version 330
#ifdef GL_ES
precision mediump float;
#endif
#define CENTER_2D vec2(0.0)
#define M_PI 3.1415926535897932384626433832795
uniform vec2    u_resolution;
uniform vec2    u_mouse;
uniform float   u_time;
out vec4 FragColor;

#include "lygia/generative/fbm.glsl"
#include "lygia/sdf/circleSDF.glsl"
#include "lygia/sdf/sphereSDF.glsl"
#include "lygia/space/rotate.glsl"
#include "lygia/generative/voronoi.glsl"
// Create crack pattern
#include "lygia/draw/fill.glsl"
vec2 normalizeCoords(vec2 uv, vec2 resolution) {
    vec2 p = uv / resolution * 2.0 - 1.0;
    p.x = p.x * resolution.x / resolution.y;
    return p;
}
float impulse( float k, float x ){
    float h = k*x;
    return h*exp(1.0-h);
}
float plot(vec2 st, float pct){
    return  smoothstep( pct-0.82, pct, st.y) -
    smoothstep( pct, pct+0.82, st.y);
}

void main(void) {
    float freq = 2.0;
    float t = u_time / freq;
    vec2 st = gl_FragCoord.xy / u_resolution;
    //vec2 st = normalizeCoords(gl_FragCoord.xy, u_resolution);
    vec2 circlePos = vec2(0.5, 0.5); // x, y position
    vec2 mouse_st = u_mouse / u_resolution;
    float y = impulse(12.,st.x);

    //float y = exp(st.x) / (exp(1.0));
    vec3 gradColor = vec3(y) + vec3(0.2, 0.2, 0.2);
    float plt = plot(st, y);
    vec4 color = vec4(gradColor, 1.0);
    float n = fbm(st * 3.0 + t);
    // Apply n to color pattern
    color = mix(color, vec4(0.1, 0.3, 0.5, 0.8), plt);
    vec2 toCenter = circlePos - st;

    float radius = 0.10;
    float sdf = sphereSDF(vec3(st - circlePos, 0.0), radius);
    float mask = fill(sdf, 0.5);
    if (length(toCenter) < radius) {
        // Simulate 3D sphere normal
        vec3 normal = normalize(vec3(toCenter, sqrt(radius * radius - dot(toCenter, toCenter))));
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
        float diffuse = max(dot(normal, lightDir), 0.0);

        // Create cloud layer using FBM noise
        // Sample noise at the sphere surface position

        vec2 noiseCoord = (st - circlePos) * 15.0; // Scale for detail
        float clouds = fbm(vec3(noiseCoord, t / 3.0)) * 2.0;
        //float clouds = fbm(vec3(noiseCoord, t)); // Animated clouds

        // Define base planet colors
        vec3 oceanColor = vec3(0.1, 0.3, 0.6); // Blue ocean
        vec3 landColor = vec3(0.3, 0.5, 0.2);  // Green land
        vec3 cloudColor = vec3(1.0, 1.0, 1.0); // White clouds

        // Create land/ocean pattern with different noise
        float terrain = fbm(vec3(noiseCoord * 0.5, 0.5)) * 5;
        vec3 baseColor = mix(oceanColor, landColor, smoothstep(0.4, 0.6, terrain));

        // Add clouds on top
        float cloudMask = smoothstep(0.5, 0.7, clouds);
        vec3 surfaceColor = mix(baseColor, cloudColor, cloudMask);

        // Apply lighting to the final surface
        vec3 litColor = surfaceColor * (diffuse * 0.8 + 0.2); // 0.2 = ambient light
//        float destructTime = fract(t * 0.3);
//        float noise = fbm(vec3((st - circlePos) * 15.0, t * 0.5));
//
//        // Fragments fly outward based on noise
//        if (noise < destructTime) {
//            discard; // Remove this pixel
//        }

        // Fade remaining pieces
        //float alpha = 1.0 - smoothstep(destructTime - 0.2, destructTime, noise);
        color.rgb += litColor * mask;
        //color.rgb += litColor * mask;
    }

//    if (length(toCenter) < radius) {
//        // Simulate 3D sphere normal
////        vec3 normal = normalize(vec3(toCenter, sqrt(radius * radius - dot(toCenter, toCenter))));
////        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
////        float diffuse = max(dot(normal, lightDir), 0.0);
////        vec3 circleColor = vec3(0.8, 0.5, 0.3) * diffuse * mask;
////
////        color.rgb += circleColor;
//
//    }
//    float lighting = (st.y + 1.0) * 0.3; // vertical gradient
//
//    float mask = fill(sdf, 0.5);
    //color.rgb += sin(n * M_PI);



    FragColor = color;
}