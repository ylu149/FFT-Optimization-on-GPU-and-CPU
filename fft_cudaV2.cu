#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuComplex.h>
#include <time.h>
#include <windows.h>

#define PI 3.14159265358979323846

struct timespec start, end;

__global__ void FFT_kernel(float2 *d_data, int N, int m, int k)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x; //j Loop controlled by tid
    if (j < m/2)
    {
        int t = j + k;
        int u = t + m/2;
        // printf("s: %d, k: %d, j: %d\n", (int)log2f(m), k, j);
        float2 twiddle_factor = make_float2(cosf(2 * PI * j / m), sinf(2 * PI * j / m));
        float2 temp = cuCmulf(d_data[u], twiddle_factor);
        d_data[u] = cuCsubf(d_data[t], temp);
        d_data[t] = cuCaddf(d_data[t], temp);
    }
}


void FFT(float2 *h_data, int N)
{
    float2 *d_data;
    cudaMalloc((void **)&d_data, sizeof(float2) * N);
    cudaMemcpy(d_data, h_data, sizeof(float2) * N, cudaMemcpyHostToDevice);
    int log2n = log2f(N);
    int threads_per_block = 256;
    int num_blocks = (N - 1) / threads_per_block + 1;

    for (int s = 1; s <= log2n; s++)
    {
        int m = 1 << s;
        for (int k = 0; k < N; k += m)
        {
            FFT_kernel<<<num_blocks, N/2>>>(d_data, N, m, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(h_data, d_data, sizeof(float2) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

int main()
{
    int N = 8;
    float2 *h_data = (float2 *)malloc(sizeof(float2) * N);
    for (int i = 0; i < N; i++)
    {
        h_data[i] = make_float2(i, 0);
    }
    // for (int i = 0; i < N; i++)
    // {
    //     printf("(%f, %f)\n", h_data[i].x, h_data[i].y);
    // }

    // clock_gettime(CLOCK_REALTIME, &start);
    // clock_gettime(CLOCK_REALTIME, &end);
    // printf("\nTime: %f\n", interval(start,end));

    LARGE_INTEGER start, end, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start); // Record the start time
    FFT(h_data, N);
    QueryPerformanceCounter(&end);   // Record the end time
    double elapsed_time = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    printf("Elapsed time: %f seconds\n", elapsed_time);

    for (int i = 0; i < N; i++)
    {
        printf("(%f, %f)\n", h_data[i].x, h_data[i].y);
    }

    free(h_data);
    return 0;
}
