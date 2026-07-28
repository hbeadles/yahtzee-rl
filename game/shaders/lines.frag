#version 330 core
#ifdef GL_ES
precision mediump float;
#endif
#include "lygia/space/aspect.glsl"
#include "lygia/space/ratio.glsl"
#include "lygia/generative/random.glsl"
#include "lygia/math/map.glsl"

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;
out vec4 FragColor;
vec3 BACKGROUND_COLOR = vec3(0.3f, 0.0f, 0.0f);
vec3 LINE_COLOR = vec3(0.76, 0.22, 0.85);
float BLOCK_WIDTH = 0.03;
float MIN_V = 0.02;
float MAX_V = 0.1;

float impulse( float k, float x ){
    float h = k*x;
    return h*exp(1.0-h);
}
float plot(vec2 st, float pct, float factor){
    return  smoothstep( pct-factor, pct, st.y) -
    smoothstep( pct, pct+factor, st.y);
}

vec2 normalizeCoords(vec2 uv, vec2 resolution) {
    vec2 p = uv / resolution * 2.0 - 1.0;
    p.x = p.x * resolution.x / resolution.y;
    return p;
}

void main(){
    float freq = 2.0;
    float t = u_time / freq;
    vec3 color = vec3(0.0f);
    vec3 final_color = vec3(0.0f);
    vec3 back_color = vec3(0.0f);
    vec2 st = normalizeCoords(gl_FragCoord.xy, u_resolution);
    vec2 mouse_st = u_mouse / u_resolution;
    st.y += 0.1;
    float wave_strength = 0.0f;
    float wave_width = 0.01;
	float c1 = mod(st.x, 2.0 * BLOCK_WIDTH);
	c1 = step(BLOCK_WIDTH, c1);
	
	float c2 = mod(st.y, 2.0 * BLOCK_WIDTH);
	c2 = step(BLOCK_WIDTH, c2);
	
	back_color = mix(st.x * BACKGROUND_COLOR, st.y * LINE_COLOR, c1 * c2);
    
    for (float i = 0; i < 6; i++){
        float r = random(vec2(i, 123.45));  
        float final_r = map(r, 0.0, 1.0, MIN_V, MAX_V);
        float dir = sin(i * 0.7 - (0.03));
        wave_width = abs(1 / (150 * st.y));
        st.y += 0.3 * sin(st.x + i / 7.0 - dir * t) / 2.0f; 
        //wave_strength += plot(st, y, final_r);
        color += vec3(wave_width * 1.9, wave_width, wave_width * 0.8);
    }
    final_color = back_color * 0.3 + color;
    //final_color = mix(final_color, color, wave_strength);
    FragColor = vec4(final_color.rgb, 1.0f);
}