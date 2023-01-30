#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define COLOR_CHANNELS 1
#define BLOCK_SIZE 16
#define GRAYLEVELS 256
#define DESIRED_NCHANNELS 1

// #define LOGGER
#define PERF
#define OPTIMIZED

typedef unsigned long long ULL;
typedef unsigned long UL;
typedef unsigned char UC;

/**
 * module load CUDA/10.1.243-GCC-8.3.0
 * nvcc -o histogram.out histogram.cu
 * srun --reservation=fri --gpus=1 ./histogram.out images/500.jpg out/500.jpg
 */

#ifndef OPTIMIZED

__global__ void KERNEL_CalculateHistogram(const UC *image, const int width, const int height, ULL *histogram)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height && x < width)
    {
        int hist_index = image[y * width + x];
        atomicAdd(histogram + hist_index, 1);
    }
}

__global__ void KERNEL_CalculateCDF(ULL *histogram, ULL *cdf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, i;
    ULL sum = 0;

    for (i = 0; i <= x; ++i)
    {
        sum += histogram[i];
    }

    cdf[x] = sum;
}

__global__ void KERNEL_findMin(const ULL *cdf, ULL *min_ptr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (cdf[x] > 0)
    {
        ULL prev = ULONG_LONG_MAX;
        while (cdf[x] < prev)
        {
            prev = atomicCAS(min_ptr, prev, cdf[x]);
        }
    }
}

#else

__global__ void KERNEL_CalculateHistogram(const UC *image, const int width, const int height, ULL *histogram)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height && x < width)
    {
        int hist_index = image[y * width + x];
        atomicAdd(histogram + hist_index, 1);
    }
}

__global__ void KERNEL_CalculateCDF(ULL *histogram, ULL *cdf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, i;
    cdf[x] = histogram[x];

    for (i = 1; i < GRAYLEVELS; i *= 2)
    {
        if (threadIdx.x >= i)
            cdf[threadIdx.x] += cdf[threadIdx.x - i];
    }
}

__global__ void KERNEL_findMin(const ULL *cdf, ULL *min_ptr)
{
    __shared__ ULL shared[GRAYLEVELS];
    int x = blockIdx.x * blockDim.x + threadIdx.x, i;
    shared[threadIdx.x] = cdf[x] > 0 ? cdf[x] : ULONG_LONG_MAX;
    __syncthreads();

    for (i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i && shared[threadIdx.x] > shared[threadIdx.x + i])
            shared[threadIdx.x] = shared[threadIdx.x + i];

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        ULL prev = ULONG_LONG_MAX;
        while (*shared < prev)
            prev = atomicCAS(min_ptr, prev, *shared);
    }
}

#endif

__device__ inline UC scale(UL cdf, UL cdfmin, UL imageSize)
{
    float scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    return (int)round(scale * (float)(GRAYLEVELS - 1));
}

__global__ void KERNEL_Equalize(const UC *image_in, UC *image_out, const int width, const int height, const ULL *cdf, const ULL *cdfmin)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const UL imageSize = width * height;

    if (y < height && x < width)
    {
        image_out[y * width + x] = scale(cdf[image_in[y * width + x]], *cdfmin, imageSize);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Error: Missing 2 image params\n");
        exit(EXIT_FAILURE);
    }

    // Read image from file
    int width, height, cpp;

    // read only DESIRED_NCHANNELS channels from the input image:
    UC *imageIn = stbi_load(argv[1], &width, &height, &cpp, DESIRED_NCHANNELS);
    if (imageIn == NULL)
    {
        printf("Error: loading image\n");
        return 1;
    }

    #ifdef LOGGER
        printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);
    #endif

    const size_t img_size = width * height * sizeof(UC);
    const size_t hist_size = GRAYLEVELS * sizeof(ULL);
    const size_t ull_size = sizeof(ULL);

    // Allocate memory for raw output image data
    UC *imageOut = (UC *)malloc(img_size);

    // Allocate memory for cuda
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(ceil(width / blockSize.x), ceil(height / blockSize.y));

    UC *d_imageIn;
    UC *d_imageOut;
    ULL *d_histogram;
    ULL *d_cdf;
    ULL *d_cdfmin;
    ULL max_value = ULONG_LONG_MAX;

    checkCudaErrors(cudaMalloc(&d_imageIn, img_size));
    checkCudaErrors(cudaMalloc(&d_imageOut, img_size));
    checkCudaErrors(cudaMalloc(&d_histogram, hist_size));
    checkCudaErrors(cudaMalloc(&d_cdf, hist_size));
    checkCudaErrors(cudaMalloc(&d_cdfmin, ull_size));

    // Copy image CPU -> CUDA and set every cell in histogram and cdf to 0
    checkCudaErrors(cudaMemcpy(d_imageIn, imageIn, img_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, hist_size));
    checkCudaErrors(cudaMemset(d_cdf, 0, hist_size));
    checkCudaErrors(cudaMemcpy(d_cdfmin, &max_value, ull_size, cudaMemcpyHostToDevice));

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Histogram equalization steps:
    cudaEventRecord(start);

    // 1. Create the histogram for the input grayscale image.
    KERNEL_CalculateHistogram<<<gridSize, blockSize>>>(d_imageIn, width, height, d_histogram);

    // 2. Calculate the cumulative distribution histogram.
    dim3 blocks256(GRAYLEVELS);
    dim3 gridSizeHist(1);
    KERNEL_CalculateCDF<<<gridSizeHist, blocks256>>>(d_histogram, d_cdf);

    // 3. Calculate the OPTIMIZED gray-level values through the general histogram equalization formula and assign OPTIMIZED pixel values
    dim3 gridSizeMin((GRAYLEVELS + blocks256.x - 1) / blocks256.x);
    KERNEL_findMin<<<gridSizeMin, blocks256>>>(d_cdf, d_cdfmin);

    #ifdef LOGGER
        checkCudaErrors(cudaMemcpy(&max_value, d_cdfmin, ull_size, cudaMemcpyDeviceToHost));
        printf("First greater than 0: %llu\n", max_value);
    #endif

    KERNEL_Equalize<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, d_cdf, d_cdfmin);
    cudaEventRecord(stop);

    // Copy data CUDA -> CPU
    checkCudaErrors(cudaMemcpy(imageOut, d_imageOut, img_size, cudaMemcpyDeviceToHost));

    // Wait for the event to finish
    cudaEventSynchronize(stop);

    #ifdef PERF
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("%0.3f\n", milliseconds);
    #endif

    // Create output image:
    stbi_write_jpg(argv[2], width, height, DESIRED_NCHANNELS, imageOut, 100);
    // stbi_write_png("out.png", width, height, DESIRED_NCHANNELS, imageOut, width * DESIRED_NCHANNELS);

    // Free CUDA memory
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_imageOut));
    checkCudaErrors(cudaFree(d_histogram));
    checkCudaErrors(cudaFree(d_cdf));
    checkCudaErrors(cudaFree(d_cdfmin));

    // Clean up the two events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free memory
    free(imageIn);
    free(imageOut);

    return 0;
}
