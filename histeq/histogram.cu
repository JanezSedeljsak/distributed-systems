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

unsigned long findMin(unsigned long long *cdf)
{

    unsigned long min = 0;
    for (int i = 0; min == 0 && i < GRAYLEVELS; i++)
    {
        min = cdf[i];
    }

    return min;
}

__device__ inline unsigned char scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize)
{
    float scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    return (int)round(scale * (float)(GRAYLEVELS - 1));
}

__global__ void KERNEL_CalculateHistogram(const unsigned char *image, const int width, const int height, unsigned long long *histogram)
{
    // Get pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int hist_index = image[y * width + x];
    atomicAdd(histogram + hist_index, 1);
}

void CalculateCDF(unsigned long long *histogram, unsigned long long *cdf)
{

    // clear cdf:
    for (int i = 0; i < GRAYLEVELS; i++)
    {
        cdf[i] = 0;
    }

    // calculate cdf from histogram
    cdf[0] = histogram[0];
    for (int i = 1; i < GRAYLEVELS; i++)
    {
        cdf[i] = cdf[i - 1] + histogram[i];
    }
}

__global__ void KERNEL_Equalize(const unsigned char *image_in, unsigned char *image_out, const int width, const int height, const unsigned long long *cdf, const unsigned long cdfmin)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const  unsigned long imageSize = width * height;

    image_out[y * width + x] = scale(cdf[image_in[y * width + x]], cdfmin, imageSize);
}

int main(int argc, char **argv)
{
    if (argc < 1)
    {
        printf("Error: Missing input image param\n");
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

    // Allocate memory for raw output image data, histogram, and CDF
    unsigned char *imageOut = (unsigned char *)malloc(img_size);
    unsigned long long *histogram = (unsigned long long *)malloc(hist_size);
    unsigned long long *CDF = (unsigned long long *)malloc(hist_size);

    // Allocate memory for cuda
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(ceil(width / blockSize.x), ceil(height / blockSize.y));

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;
    unsigned long long *d_histogram;
    unsigned long long *d_cdf;

    checkCudaErrors(cudaMalloc(&d_imageIn, img_size));
    checkCudaErrors(cudaMalloc(&d_imageOut, img_size));
    checkCudaErrors(cudaMalloc(&d_histogram, hist_size));
    checkCudaErrors(cudaMalloc(&d_cdf, hist_size));

    // Copy image CPU -> CUDA and set every cell in histogram and cdf to 0
    checkCudaErrors(cudaMemcpy(d_imageIn, imageIn, img_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, hist_size));
    checkCudaErrors(cudaMemset(d_cdf, 0, hist_size));

    // Histogram equalization steps:
    // 1. Create the histogram for the input grayscale image.
    KERNEL_CalculateHistogram<<<gridSize, blockSize>>>(d_imageIn, width, height, d_histogram);
    checkCudaErrors(cudaMemcpy(histogram, d_histogram, hist_size, cudaMemcpyDeviceToHost));

    // 2. Calculate the cumulative distribution histogram.
    // TODO
    CalculateCDF(histogram, CDF);
    checkCudaErrors(cudaMemcpy(d_cdf, CDF, hist_size, cudaMemcpyHostToDevice));

    // 3. Calculate the new gray-level values through the general histogram equalization formula
    //    and assign new pixel values
    unsigned long cdfmin = findMin(CDF); // todo make reduction
    KERNEL_Equalize<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, d_cdf, cdfmin);

    // Copy data CUDA -> CPU
    checkCudaErrors(cudaMemcpy(imageOut, d_imageOut, img_size, cudaMemcpyDeviceToHost));

    // Create output image:
    // stbi_write_png("out.png", width, height, DESIRED_NCHANNELS, imageOut, width * DESIRED_NCHANNELS);
    stbi_write_jpg("kolesarko2.jpg", width, height, DESIRED_NCHANNELS, imageOut, 100);

    // Free CUDA memory
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_imageOut));
    checkCudaErrors(cudaFree(d_histogram));
    checkCudaErrors(cudaFree(d_cdf));

    // Free memory
    free(imageIn);
    free(imageOut);
    free(histogram);
    free(CDF);

    return 0;
}
