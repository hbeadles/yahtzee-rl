#version 330 core
#ifdef GL_ES
precision mediump float;
#endif

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;
out vec4 FragColor;

void main(){
    vec2 st = gl_FragCoord.xy/u_resolution;
    vec2 mouse_st = u_mouse / u_resolution;
    float t = u_time / 2.0;
    float dist = distance(st, mouse_st);
    float circle = smoothstep(0.1, 0.05, dist);

    vec3 color = vec3(abs(sin(st.x - t)), abs(cos(st.y - t)), abs(sin(t)));
    color = mix(color, color * 1.8, circle);
    FragColor = vec4(color, 1.0);
}