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

/**
 * module load CUDA/10.1.243-GCC-8.3.0
 * nvcc -o histogram.out histogram.cu
 * srun --reservation=fri --gpus=1 ./histogram.out kolesar-neq.jpg
 */

// TODO: add reduction and get actual min value of cdf array and store it into min_ptr
__global__ void KERNEL_findMin(unsigned long long *cdf, unsigned long long *min_ptr)
{
    *min_ptr = 29;
}

// OK
__device__ inline unsigned char scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize)
{
    float scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    return (int)round(scale * (float)(GRAYLEVELS - 1));
}

// TODO: Add shared memory
__global__ void KERNEL_CalculateHistogram(const unsigned char *image, const int width, const int height, unsigned long long *histogram)
{
    // Get pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int hist_index = image[y * width + x];
    atomicAdd(histogram + hist_index, 1);
}

// TODO: improve this
__global__ void KERNEL_CalculateCDF(unsigned long long *histogram, unsigned long long *cdf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long sum = 0;

    for (int i = 0; i <= x; i++)
    {
        sum += histogram[i];
    }

    cdf[x] = sum;
}

// OK
__global__ void KERNEL_Equalize(const unsigned char *image_in, unsigned char *image_out, const int width, const int height, const unsigned long long *cdf, const unsigned long long *cdfmin)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long imageSize = width * height;

    image_out[y * width + x] = scale(cdf[image_in[y * width + x]], *cdfmin, imageSize);
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
    unsigned char *imageIn = stbi_load(argv[1], &width, &height, &cpp, DESIRED_NCHANNELS);
    if (imageIn == NULL)
    {
        printf("Error: loading image\n");
        return 1;
    }
    printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);

    const size_t img_size = width * height * sizeof(unsigned char);
    const size_t hist_size = GRAYLEVELS * sizeof(unsigned long long);

    // Allocate memory for raw output image data
    unsigned char *imageOut = (unsigned char *)malloc(img_size);

    // Allocate memory for cuda
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(ceil(width / blockSize.x), ceil(height / blockSize.y));

    dim3 blockSizeHist(1);
    dim3 gridSizeHist(GRAYLEVELS);

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;
    unsigned long long *d_histogram;
    unsigned long long *d_cdf;
    unsigned long long *d_cdfmin;

    checkCudaErrors(cudaMalloc(&d_imageIn, img_size));
    checkCudaErrors(cudaMalloc(&d_imageOut, img_size));
    checkCudaErrors(cudaMalloc(&d_histogram, hist_size));
    checkCudaErrors(cudaMalloc(&d_cdf, hist_size));
    checkCudaErrors(cudaMalloc(&d_cdfmin, sizeof(unsigned long long)));

    // Copy image CPU -> CUDA and set every cell in histogram and cdf to 0
    checkCudaErrors(cudaMemcpy(d_imageIn, imageIn, img_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, hist_size));
    checkCudaErrors(cudaMemset(d_cdf, 0, hist_size));

    // Histogram equalization steps:

    // 1. Create the histogram for the input grayscale image.
    KERNEL_CalculateHistogram<<<gridSize, blockSize>>>(d_imageIn, width, height, d_histogram);

    // 2. Calculate the cumulative distribution histogram.
    KERNEL_CalculateCDF<<<gridSizeHist, blockSizeHist>>>(d_histogram, d_cdf);

    // 3. Calculate the new gray-level values through the general histogram equalization formula and assign new pixel values
    KERNEL_findMin<<<blockSizeHist, gridSizeHist>>>(d_cdf, d_cdfmin);
    KERNEL_Equalize<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, d_cdf, d_cdfmin);

    // Copy data CUDA -> CPU
    checkCudaErrors(cudaMemcpy(imageOut, d_imageOut, img_size, cudaMemcpyDeviceToHost));

    // Create output image:
    // stbi_write_png("out.png", width, height, DESIRED_NCHANNELS, imageOut, width * DESIRED_NCHANNELS);
    stbi_write_jpg(argv[2], width, height, DESIRED_NCHANNELS, imageOut, 100);

    // Free CUDA memory
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_imageOut));
    checkCudaErrors(cudaFree(d_histogram));
    checkCudaErrors(cudaFree(d_cdf));
    checkCudaErrors(cudaFree(d_cdfmin));

    // Free memory
    free(imageIn);
    free(imageOut);

    return 0;
}
