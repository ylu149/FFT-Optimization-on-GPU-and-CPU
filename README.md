# FFT Multithreading on CPU & GPU

## Overview
This project implements Fast Fourier Transforms (FFT) using the Cooley-Tukey algorithm, optimized for both CPU and GPU. It aims to demonstrate efficient multithreading techniques and the benefits of parallel computing in FFT calculations.

## Features
- **Serial FFT Implementation**: Utilizes the Cooley-Tukey algorithm for basic FFT calculations.
- **CPU Multithreading**: Implements FFT using Pthreads to leverage CPU cores for improved performance.
- **GPU Accelerated FFT**: Employs GPU capabilities for high-speed FFT processing.
- **Performance Analysis**: Includes comparisons between serial, multithreaded CPU, and GPU implementations.
- **Code Extensibility**: Structured for easy modification and extension for further research and optimization.
  
## Key Files
- **fft_final.c**: This is the main C file for the CPU-based FFT implementation. It includes the various serial versions of the FFT algorithm as well as the multithreaded version using Pthreads.
- **fft_cudaV2.cu**: This CUDA file contains the GPU-accelerated version of the FFT, optimized for performance on NVIDIA GPUs. It demonstrates the use of CUDA for parallelizing FFT calculations.
- **cuda_fft2d.cu**: Focused on 2D FFT calculations, this CUDA file provides an implementation specifically optimized for 2D data structures, taking full advantage of GPU parallel processing capabilities.

## Installation
1. Clone the repository ``https://github.com/ylu149/FFT-Optimization-on-GPU-and-CPU.git``
2. Navigate to the project directory: `cd FFT-Multithreading-CPU-GPU`
3. Get a compatible Cuda and c++ compiler and choose the from the following (note: lpthread and math libraries may be needed):

## Usage
- **fft_final.c**: `gcc fft_final.c -o fft_final -lpthread -lm`
- **fft_cudaV2.cu**: `nvcc -o fft_cudaV2.cu -lcufft`
- **cuda_fft2d.cu**: `nvcc -o cuda_fft2d.cu -lcufft`
