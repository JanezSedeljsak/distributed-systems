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

__global__ void KERNEL_findMin(unsigned long long *cdf, unsigned long *min_ptr)
{
    int block = blockIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long current = cdf[x];
    if (x == 0 && block == 0)
    {
        *min_ptr = current;
    }

    __syncthreads();
    for (int i = 1; i < blockDim.x * gridDim.x; i++)
    {
        if (x + i < GRAYLEVELS)
        {
            current = cdf[x + i] < current ? cdf[x + i] : current;
            if (current == cdf[x + i])
            {
                *min_ptr = current;
            }
        }
        __syncthreads();
    }
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

__global__ void KERNEL_Equalize(const unsigned char *image_in, unsigned char *image_out, const int width, const int height, const unsigned long long *cdf, const unsigned long cdfmin)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long imageSize = width * height;

    image_out[y * width + x] = scale(cdf[image_in[y * width + x]], cdfmin, imageSize);
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

    // Allocate memory for raw output image data, histogram, and CDF
    unsigned char *imageOut = (unsigned char *)malloc(img_size);
    unsigned long long *CDF = (unsigned long long *)malloc(hist_size);

    // Allocate memory for cuda
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(ceil(width / blockSize.x), ceil(height / blockSize.y));

    dim3 blockSizeHist(1);
    dim3 gridSizeHist(GRAYLEVELS);

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

    // 2. Calculate the cumulative distribution histogram.
    KERNEL_CalculateCDF<<<gridSizeHist, blockSizeHist>>>(d_histogram, d_cdf);
    

    // 3. Calculate the new gray-level values through the general histogram equalization formula
    //    and assign new pixel values
    checkCudaErrors(cudaMemcpy(CDF, d_cdf, hist_size, cudaMemcpyDeviceToHost));
    unsigned long cdfmin = findMin(CDF);
    // unsigned long cdfmin;
    // KERNEL_findMin<<<gridSize, blockSize>>>(d_cdf, &cdfmin);
    KERNEL_Equalize<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, d_cdf, cdfmin);

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

    // Free memory
    free(imageIn);
    free(imageOut);
    free(CDF);

    return 0;
}
