vec2 normalizeCoords(vec2 uv, vec2 resolution) {
    vec2 p = uv / resolution * 2.0 - 1.0;
    p.x = p.x * resolution.x / resolution.y;
    return p;
}

float plot(vec2 st, float pct, float factor){
    return  smoothstep( pct-factor, pct, st.y) -
    smoothstep( pct, pct+factor, st.y);
}

float BLOCK_WIDTH = 0.05;
vec3 BACKGROUND_COLOR = vec3(0.2f, 0.2f, 0.5f);
vec3 LINE_COLOR = vec3(0.73, 0.0, 0.42);
vec3 OUTPUT_LINE = vec3(0.45, 0.0, 0.32);
int NUM_LINES = 4;

float random(in vec2 st) {
#ifdef RANDOM_SINLESS
    vec3 p3  = fract(vec3(st.xyx) * RANDOM_SCALE.xyz);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
#else
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
#endif
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float freq = 2.0;
    float t = iTime / freq;

    vec2 st = normalizeCoords(fragCoord, iResolution.xy);

    // --- background tint ---
    vec3 back_color = BACKGROUND_COLOR * vec3(sin(t) + 0.1, 0.0, cos(t) + 0.1);

    // --- Layer 1: grid (your original expression) ---
    float c1 = step(BLOCK_WIDTH, mod(st.x, 2.0 * BLOCK_WIDTH));
    float c2 = step(BLOCK_WIDTH, mod(st.y, 2.0 * BLOCK_WIDTH));
    vec3 col = mix(st.x * back_color, st.y * LINE_COLOR, c1 * c2);

    // --- Layer 2: plotted curve, composited on top ---
    float spacing = 0.1;   // vertical gap between lines
    float phase   = 0.6;    // how much each line lags the previous one
    for (int i = 0; i < NUM_LINES; i++) {
        float fi = float(i);
        vec3 line_color = OUTPUT_LINE + vec3((fi + 0.1), 0.0, (fi + 0.1));
        float offset = (fi - float(NUM_LINES - 1) * 0.3) * spacing;
        float pitch = random(vec2(i, 10));
        float increment = random(vec2(0.1, 0.8));
        float speed = 1.0 + fi * 0.15;   // each line advances at a different rate
        float y = increment * sin(st.x * pitch + t * speed) + offset;

        float line = plot(st, y, 0.02);
        col = mix(col, line_color, line);
    }

    fragColor = vec4(col, 1.0);
}