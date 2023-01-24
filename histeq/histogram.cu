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

unsigned char scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize)
{

    float scale;

    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);

    scale = round(scale * (float)(GRAYLEVELS - 1));

    return (int)scale;
}

__global__ void KERNEL_CalculateHistogram(const unsigned char *image, const int width, const int height, unsigned long long *histogram)
{
    // Get pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int hist_index = image[y * width + x];
    atomicAdd(histogram + hist_index, 1);
}

void CalculateHistogram(unsigned char *image, int width, int height, unsigned long long *histogram)
{

    // Clear histogram:
    for (int i = 0; i < GRAYLEVELS; i++)
    {
        histogram[i] = 0;
    }

    // Calculate histogram
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            histogram[image[i * width + j]]++;
        }
    }
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

void Equalize(unsigned char *image_in, unsigned char *image_out, int width, int height, unsigned long long *cdf)
{

    unsigned long imageSize = width * height;

    unsigned long cdfmin = findMin(cdf);

    // Equalize
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            image_out[(i * width + j)] = scale(cdf[image_in[i * width + j]], cdfmin, imageSize);
        }
    }
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

    // Copy image CPU -> CUDA
    checkCudaErrors(cudaMemcpy(d_imageIn, imageIn, img_size, cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_imageOut, imageIn, img_size, cudaMemcpyHostToDevice));

    // Histogram equalization steps:
    // 1. Create the histogram for the input grayscale image.
    // CalculateHistogram(imageIn, width, height, histogram);
    KERNEL_CalculateHistogram<<<gridSize, blockSize>>>(d_imageIn, width, height, d_histogram);
    checkCudaErrors(cudaMemcpy(histogram, d_histogram, hist_size, cudaMemcpyDeviceToHost));

    // 2. Calculate the cumulative distribution histogram.
    CalculateCDF(histogram, CDF);
    // 3. Calculate the new gray-level values through the general histogram equalization formula
    //    and assign new pixel values
    Equalize(imageIn, imageOut, width, height, CDF);

    // // Copy data CUDA -> CPU
    // checkCudaErrors(cudaMemcpy(histogram, d_histogram, hist_size, cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(CDF, d_cdf, hist_size, cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(imageOut, d_imageOut, img_size, cudaMemcpyDeviceToHost));

    // Create output image:
    // stbi_write_png("out.png", width, height, DESIRED_NCHANNELS, imageOut, width * DESIRED_NCHANNELS);
    stbi_write_jpg("kolesarko.jpg", width, height, DESIRED_NCHANNELS, imageOut, 100);

    // Free cuda memory
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
