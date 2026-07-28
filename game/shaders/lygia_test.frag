#version 330 core

#include "lygia/generative/snoise.glsl"
#include "lygia/generative/fbm.glsl"
#include "lygia/color/palette.glsl"

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

out vec4 FragColor;

void main() {
    vec2 st = gl_FragCoord.xy / u_resolution;
    vec2 mouse_st = u_mouse / u_resolution;

    // Animated noise
    float n = fbm(vec3(st * 3.0, u_time * 0.1));

    // Distance from mouse
    float dist = distance(st, mouse_st);

    // Color palette based on noise and distance
    vec3 color = palette(n + dist,
    vec3(0.5),
    vec3(0.5),
    vec3(1.0),
    vec3(0.0, 0.33, 0.67));

    FragColor = vec4(color, 1.0);
}