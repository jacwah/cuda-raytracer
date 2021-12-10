#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>

#define BG_COLOR (float3{0.0f, 0.0f, 0.0f})

//#define SPHERE_CENTER (float3{0.0f, 0.0f, 1.0f})
#define SPHERE_CENTER (float3{0.0f, 0.0f, 1.0f})
#define SPHERE_RADIUS 1.0f
#define FG_COLOR (float3{0.0f, 0.0f, 1.0f})
#define DIFFUSE 1.0f
#define SPECULAR_C 1.0f
#define SPECULAR_K 50

#define LIGHT_POS (float3{5.0f, 5.0f, -10.0f})
#define LIGHT_COLOR (float3{1.0f, 1.0f, 1.0f})
#define AMBIENT_COLOR (float3{0.05f, 0.05f, 0.05f})

#define CAMERA_POS (float3{0.0f, 0.0f, -1.0f})
#define CAMERA_FOCUS (float3{0.0f, 0.0f, 0.0f})

#define NUM_STREAMS 2

__device__ float3 operator*(float a, float3 v)
{
    return {a*v.x, a*v.y, a*v.z};
}

__device__ float3 operator+(float3 v, float3 w)
{
    return {v.x+w.x, v.y+w.y, v.z+w.z};
}

__device__ float3 operator+=(float3& v, float3 w)
{
    v.x += w.x;
    v.y += w.y;
    v.z += w.z;
    return v;
}

__device__ float3 operator-(float3 v, float3 w)
{
    return {v.x-w.x, v.y-w.y, v.z-w.z};
}

__device__ float3 normalize(float3 v)
{
    float rn = rnorm3df(v.x, v.y, v.z);
    return rn*v;
}

__device__ float dot(float3 v, float3 w)
{
    return v.x*w.x + v.y*w.y + v.z*w.z;
}

__device__ float3 clamp(float3 v)
{
    return {__saturatef(v.x), __saturatef(v.y), __saturatef(v.z)};
}

__device__ float intersect_sphere(float3 origin, float3 dir, float3 center, float radius)
{
    float3 os = origin - center;
    float b = 2.0f * dot(dir, os);
    float c = dot(os, os) - radius*radius;
    float d = b*b - 4*c;
    if (d > 0.0f) {
        float ds = sqrtf(d);
        float q;
        if (b < 0.0f)
            q = (-b - ds) / 2.0f;
        else
            q = (-b + ds) / 2.0f;
        float t0 = q;
        float t1 = c / q;
        if (t0 > t1) {
            float t2 = t0;
            t0 = t1;
            t1 = t2;
        }
        if (t1 >= 0.0f) {
            if (t0 < 0.0f)
                return t1;
            else
                return t0;
        }
    }
    return INFINITY;
}

__device__ float3 trace_ray(float3 origin, float3 dir)
{
    float t = intersect_sphere(origin, dir, SPHERE_CENTER, SPHERE_RADIUS);
    if (isinf(t))
        return BG_COLOR;

    float3 intersect = origin + t*dir;
    float3 n = normalize(intersect - SPHERE_CENTER);
    float3 toL = normalize(LIGHT_POS - intersect);
    float3 toO = normalize(origin - intersect);

    float3 color = AMBIENT_COLOR;

    float diffusity = dot(n, toL);
    if (diffusity > 0.0f)
        color += DIFFUSE*diffusity*FG_COLOR;

    float specularity = dot(n, normalize(toL + toO));
    if (specularity > 0.0f)
        color += SPECULAR_C*powf(specularity, SPECULAR_K)*LIGHT_COLOR;

    return clamp(color);
}

__global__ void render_image(uint8_t *image, unsigned w, unsigned h, unsigned line_off)
{
    const unsigned i = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned j = blockDim.y*(blockIdx.y+line_off) + threadIdx.y;

    if (i >= w || j >= h)
        return;

    float x =  2.0f*(i - (w-1)/2.0f) / (w-1.0f);
    float y = -2.0f*(j - (h-1)/2.0f) / (h-1.0f);

    float3 focus = CAMERA_FOCUS;
    focus.x = x;
    focus.y = y;
    float3 dir = normalize(focus - CAMERA_POS);

    float3 color = trace_ray(CAMERA_POS, dir);

    size_t idx = j*w + i;
    image[3*idx+0] = 255.99f*color.x;
    image[3*idx+1] = 255.99f*color.y;
    image[3*idx+2] = 255.99f*color.z;
}

double get_time()
{
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + 1e-9*now.tv_nsec;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s w h fname block\n", argv[0]);
        return 1;
    }

    unsigned w = atoi(argv[1]);
    unsigned h = atoi(argv[2]);
    const char *fname = argv[3];
    unsigned block = atoi(argv[4]);

    if (block*block > 1024) {
        fprintf(stderr, "block too large\n");
        return 1;
    }

    double start = get_time();

    FILE *fp = fopen(fname, "w");
    if (!fp) {
        perror("error opening file");
        return 1;
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "%lu %lu\n", w, h);
    fprintf(fp, "255\n", w, h);

    unsigned lines_per_chunk = 1;
    size_t chunk_size = 3UL*block*w*lines_per_chunk;

    fprintf(stderr, "chunk size: %lu\n", chunk_size);

    size_t image_size = 3UL*w*h;
    size_t written = 0;
    uint8_t *chunk_buf;
    cudaMallocHost(&chunk_buf, NUM_STREAMS*chunk_size);

    uint8_t *image_d;
    cudaMalloc(&image_d, image_size);

    dim3 block_size(block, block);
    dim3 grid((w+block-1)/block, lines_per_chunk);
    unsigned image_lines = (h+block-1)/block;

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        render_image<<<grid, block_size, 0, streams[i]>>>(image_d, w, h, i*lines_per_chunk);
        cudaMemcpyAsync(&chunk_buf[i*chunk_size], &image_d[i*chunk_size], chunk_size, cudaMemcpyDefault, streams[i]);
    }
    unsigned next_line = lines_per_chunk*NUM_STREAMS;
    unsigned next_stream = 0;

    double after_init = get_time();

    while (written < image_size) {
        cudaStreamSynchronize(streams[next_stream]);

        size_t write_size = chunk_size;
        if (written + write_size > image_size)
            write_size = image_size - written;
        size_t write = fwrite(&chunk_buf[next_stream*chunk_size], 1, write_size, fp);
        written += write_size;

        if (write != write_size) {
            perror("fwrite");
            return 1;
        }

        if (next_line < image_lines) {
            render_image<<<grid, block_size, 0, streams[next_stream]>>>(image_d, w, h, next_line);
            cudaMemcpyAsync(&chunk_buf[next_stream*chunk_size], &image_d[(next_line/lines_per_chunk)*chunk_size], chunk_size, cudaMemcpyDefault, streams[next_stream]);
            next_line += lines_per_chunk;
        }

        next_stream = (next_stream+1) % NUM_STREAMS;
    }

    double after_exec = get_time();

    fclose(fp);

    double end = get_time();

    printf("Total time:  %e\n", end-start);
    printf("---------------------\n");
    printf("init:  %e\n", after_init-start);
    printf("exec:  %e\n", after_exec-after_init);
    printf("write: %e\n", end-after_exec);
}
