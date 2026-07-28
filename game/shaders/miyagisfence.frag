#version 330 core
#ifdef GL_ES
precision mediump float;
#endif

#define PI 3.14159265359
#define WAVE_AMP 0.43
//Number of turbulence waves
#define TURB_NUM 10.0
//Turbulence wave amplitude
#define TURB_AMP 0.7
//Turbulence wave speed
#define TURB_SPEED 0.3
//Turbulence frequency
#define TURB_FREQ 2.0
//Turbulence frequency multiplier
#define TURB_EXP 1.4
uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

out vec4 FragColor;

//float plot(vec2 st, float pct){
//    return  smoothstep( pct-0.02, pct, st.y) -
//    smoothstep( pct, pct+0.02, st.y);
//}

vec2 turbulence(vec2 p)
{
    //Turbulence starting scale
    float freq = TURB_FREQ;

    //Turbulence rotation matrix
    mat2 rot = mat2(0.6, -0.8, 0.8, 0.6);

    //Loop through turbulence octaves
    for(float i=0.0; i<TURB_NUM; i++)
    {
        //Scroll along the rotated y coordinate
        float phase = freq * (p * rot).y + TURB_SPEED*u_time + i;
        //Add a perpendicular sine wave offset
        p += TURB_AMP * rot[0] * sin(phase) / freq;

        //Rotate for the next octave
        rot *= mat2(0.6, -0.8, 0.8, 0.6);
        //Scale down for the next octave
        freq *= TURB_EXP;
    }

    return p;
}

void main() {
    vec2 st = 2.0*(gl_FragCoord.xy*2.0-u_resolution.xy)/u_resolution.y;

    float freq = 2.0;
    float t = u_time / freq;
    vec2 mouse_st = u_mouse / u_resolution;
    vec2 p = turbulence(st);
    vec3 col = 0.5*exp(0.1*p.x*vec3(1,1,2));
    col /= dot(cos(p*3.534),sin(-p.yx*3.*.618))+2.0;

    FragColor = vec4(col, 1.0);
//    float y = pow(st.x, 5.0);
//    //float y = smoothstep(0.2,0.5,st.x) - smoothstep(0.5,0.8,st.x);
//
//    vec3 color = vec3(y);
//    float pct = plot(st, y);
//    color = (1.0-pct)*color+pct*vec3(0.0,1.0,0.0);
//    FragColor = vec4(color, 1.0);
}