# FFT-Optimization
The goal of this project is to create various fast fourier transform algorithms utilizing hardware optimizations via the CPU and GPU to speed up performance. 

Note that for all codes below the Radix-2 decimation in frequency Cooley-Tukey algorithmn was used as the primary algorithm for optimization. The bitreversal step was removed for the sake of evaluating just the FFT performance. 

fft_final.c contains 4 versions of FFT code. It contains the serial, multi-threading per calculation(j), multi-threading per block of calculations(k), and finally the fully optimized version that multi-threads per block of calculations(k) with s being calculated by each thread. 

cuda_fft2d.cu contains the 2D FFT algorithm for GPU. 

fft_cudaV2.cu is the ineffective GPU implementation of threading per calculation(j)

A more detailed explanation of the codes above can be found in the final presentation pdf and the final report document. 
