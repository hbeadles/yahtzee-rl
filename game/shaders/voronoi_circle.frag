#version 330 core
#ifdef GL_ES
precision mediump float;
#endif

// Uniforms can be used to control the circle size and center
uniform vec2 u_resolution;
uniform float u_time;
uniform vec2 u_mouse;
out vec4 FragColor;

// A simple hash function to generate pseudo-random points in a grid cell
vec2 hash( vec2 p ) {
    p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
    return -1.0 + 2.0 * fract(sin(p)*43758.5453123);
}

// Function to calculate the Voronoi index (cell ID)
float voronoiIndex(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float min_dist = 1e10;
    float cell_index = -1.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            // Generate a random seed point within the neighbor cell
            vec2 point = hash(i + neighbor); 
            // Transform the random point from [-1, 1] range to [0, 1]
            vec2 diff = neighbor + point - f;
            float dist = dot(diff, diff);

            if (dist < min_dist) {
                min_dist = dist;
                // Encode index. Simple int index isn't easy in GLSL, 
                // we use a float approximation here for demonstration.
                // In a real use case with N points, you might store indices in a texture.
                // This hash approximation gives a unique ID for visualization.
                cell_index = dot(i + neighbor, vec2(1.0, 1000.0));
            }
        }
    }
    return cell_index;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    // Adjust aspect ratio for non-square viewports
    uv.x *= u_resolution.x / u_resolution.y; 
    vec2 mouse_st = u_mouse / u_resolution;
    float t = u_time / 2.0;


    // Define the center and radius of the bounding circle
    vec2 center = vec2(0.5 * (u_resolution.x / u_resolution.y), 0.5);
    float radius = 0.4;
    
    // Calculate distance from center to current fragment
    float dist_to_center = distance(uv, center);

    if (dist_to_center > radius) {
        // Discard fragments outside the circle
        discard; 
    } else {
        // Calculate the Voronoi index/ID for fragments inside the circle
        // Scale 'uv' to control cell density, e.g., multiply by 5.0
        float index = voronoiIndex(uv * 5.0); 

        // Use the index to assign a color for visualization
        // The index will be a float, so we use fract() and sin() to get a colorful gradient per cell
        vec3 color = vec3(fract(sin(index) * 123.456), fract(sin(index + 1.0) * 456.789), fract(sin(index + 2.0) * 789.123));
        
        FragColor = vec4(color, 1.0);
    }
}
