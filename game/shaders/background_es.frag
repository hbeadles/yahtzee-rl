#version 300 es
#ifdef GL_ES
precision mediump float;
#endif

#include "lygia/generative/snoise.glsl"
#include "lygia/generative/fbm.glsl"
#include "lygia/color/palette.glsl"
#include "lygia/space/hexTile.glsl"
#include "lygia/sdf/hexSDF.glsl"

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

out vec4 FragColor;

void main() {
    vec2 st = gl_FragCoord.xy / u_resolution;
    vec2 mouse_st = u_mouse / u_resolution;
    // Scale the st coordinate to use hexTile
    st *= 16.0;
    vec4 t = hexTile(st);
    vec2 tileUV = t.xy;
    vec2 tileID = t.zw;
    float l = hexSDF(tileUV);
    //float n = fbm(tileID + u_time * 0.06);
    //float n = (tileID.y) + u_time * 0.1; -- Move colors left to right
    float n = 0.5 * sin(u_time + random(tileID) * 6.2813);
    vec3 color = palette(n,
            vec3(0.50, 0.50, 0.50),
            vec3(0.50, 0.50, 0.50),
            vec3(1.00, 1.00, 1.00),
            vec3(0.00, 0.10, 0.20));
    //vec3 color = palette(n, vec3(0.46, 0.17, 0.74), vec3(0.75, 0.50, 0.50), vec3(1.75, 1.00, 1.00), vec3(0.54, 0.35, 0.29));
    //float effect = smoothstep(0.5, 0.45, l);
    // Diamond effect
    // float effect = fract(l) + smoothstep(0.5, 0.45, l);
    float effect = smoothstep(1.0, 0.8, l);
    color = mix(vec3(0.0), color, effect);
    // luminance = dot(color, vec3(0.299, 0.587, 0.114));
    // Rim Glow
    float rim = smoothstep(0.8, 1.0, l);
    color += rim * vec3(0.1, 0.2, 0.4);
    color *= 0.8;

    FragColor = vec4(color, 1.0);

}