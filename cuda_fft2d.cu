#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <windows.h>
#include <cufft.h>

#define BLOCK_SIZE 32 // Block size for CUDA kernel
#define PI 3.14159265358979323846 // Value of pi

__global__ void fft_kernel(float2* data, int N) {
    // Shared memory for storing intermediate results
    __shared__ float2 shared[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Compute the global indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * N + x;

    // Copy the data to the shared memory
    shared[threadIdx.y][threadIdx.x] = data[index];
    __syncthreads();

    // Perform the row-wise FFT
    for (int k = 0; k < blockDim.x; k++) {
        float2 w;
        w.x = cosf(-2.0f * PI * k / N);
        w.y = sinf(-2.0f * PI * k / N);
        float2 t;
        t.x = shared[threadIdx.y][k].x * w.x - shared[threadIdx.y][k].y * w.y;
        t.y = shared[threadIdx.y][k].x * w.y + shared[threadIdx.y][k].y * w.x;
        shared[threadIdx.y][k] = t;
    }
    __syncthreads();

    // Perform the column-wise FFT
    for (int k = 0; k < blockDim.y; k++) {
        float2 w;
        w.x = cosf(-2.0f * PI * k / N);
        w.y = sinf(-2.0f * PI * k / N);
        float2 t;
        t.x = shared[k][threadIdx.x].x * w.x - shared[k][threadIdx.x].y * w.y;
        t.y = shared[k][threadIdx.x].x * w.y + shared[k][threadIdx.x].y * w.x;
        shared[k][threadIdx.x] = t;
    }
    __syncthreads();

    // Copy the data back to the global memory
    data[index] = shared[threadIdx.y][threadIdx.x];
}

void fft(float2* data, int N) {
    int blockSize = BLOCK_SIZE;
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(N / blockSize, N / blockSize, 1);

    // Call the CUDA kernel
    fft_kernel<<<dimGrid, dimBlock>>>(data, N);
    cudaDeviceSynchronize();
}

int main() {
    // Set up the input data
    int N = 16834*2;//CHANGE THIS LINE___________________________________________
    float2* data = (float2*) malloc(N * N * sizeof(float2));
    for (int i = 0; i < N * N; i++) {
        data[i].x = i;
        data[i].y = 0;
    }

    // Call the FFT function
    LARGE_INTEGER start, end, frequency;
    double elapsed_time = 0;
    int iters = 100;
    for(int i = 0; i < iters; i++){
        float2* d_data;
        cudaMalloc((void**) &d_data, N * N * sizeof(float2));
        cudaMemcpy(d_data, data, N * N * sizeof(float2), cudaMemcpyHostToDevice);
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start); // Record the start time
        fft(d_data, N);
        QueryPerformanceCounter(&end);   // Record the end time
        elapsed_time += (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
        cudaFree(d_data);
    }
    printf("Elapsed time: %f seconds with N^2 = %ld with iters = %d\n", elapsed_time/iters, N*N, iters);

    // Copy the output data back to the host
    //cudaMemcpy(data, d_data, N * N * sizeof(float2), cudaMemcpyDeviceToHost);

    // Free the memory on the device
    //cudaFree(d_data);

    // Print the output data
    // for (int i = 0; i < N * N; i++) {
    //     printf("(%f, %f)\n", data[i].x, data[i].y);
    // }

    // Free the memory on the host
    free(data);
    return 0;
}
