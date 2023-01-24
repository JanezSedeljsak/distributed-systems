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
 * nvcc -o hist.out hist.cu
 * srun --reservation=fri --gpus=1 ./hist.out
*/

unsigned long findMin(unsigned long* cdf){
    
    unsigned long min = 0;
    for (int i = 0; min == 0 && i < GRAYLEVELS; i++) {
		min = cdf[i];
    }
    
    return min;
}

unsigned char scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize){
    
    float scale;
    
    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    
    scale = round(scale * (float)(GRAYLEVELS-1));
    
    return (int)scale;
}

__global__ void KERNEL_CalculateHistogram(const unsigned char* image, const int width, const int height, const unsigned long* histogram) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = i * width + j;
        int value = image[index];

        atomicAdd(&histogram[value], 1);
    }
}

void CalculateHistogram(unsigned char* image, int width, int heigth, unsigned long* histogram){    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(ceil(width / blockSize.x), ceil(height / blockSize.y));

    unsigned char *d_image;
    unsigned long *d_histogram;

    const size_t img_size = width * heigth * sizeof(unsigned char);
    const size_t hist_size = GRAYLEVELS * sizeof(unsigned long);

    checkCudaErrors(cudaMalloc(&d_image, img_size));
    checkCudaErrors(cudaMalloc(&d_histogram, hist_size));

    checkCudaErrors(cudaMemcpy(d_image, image, img_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, hist_size));

    KERNEL_CalculateHistogram<<<gridSize, blockSize>>>(d_image, width, height, d_histogram);
    getLastCudaError("KERNEL_CalculateHistogram() execution failed\n");

    checkCudaErrors(cudaMemcpy(histogram, d_histogram, hist_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaFree(d_histogram));
}

void CalculateCDF(unsigned long* histogram, unsigned long* cdf){
    
    // clear cdf:
    for (int i=0; i<GRAYLEVELS; i++) {
        cdf[i] = 0;
    }
    
    // calculate cdf from histogram
    cdf[0] = histogram[0];
    for (int i=1; i<GRAYLEVELS; i++) {
        cdf[i] = cdf[i-1] + histogram[i];
    }
}


void Equalize(unsigned char * image_in, unsigned char * image_out, int width, int heigth, unsigned long* cdf){
     
    unsigned long imageSize = width * heigth;
    
    unsigned long cdfmin = findMin(cdf);
    
    //Equalize
    for (int i=0; i<heigth; i++) {
        for (int j=0; j<width; j++) {
            image_out[(i*width + j)] = scale(cdf[image_in[i*width + j]], cdfmin, imageSize);
        }
    }
}


int main(void)
{
	// Read image from file
    int width, height, cpp;
    // read only DESIRED_NCHANNELS channels from the input image:
    unsigned char *imageIn = stbi_load("kolesar-neq.jpg", &width, &height, &cpp, DESIRED_NCHANNELS);
    if(imageIn == NULL) {
        printf("Error while loading the image\n");
        return 1;
    }
    printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);
    

	//Allocate memory for raw output image data, histogram, and CDF 
	unsigned char *imageOut = (unsigned char *)malloc(height * width * sizeof(unsigned long));
    unsigned long *histogram= (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));
    unsigned long *CDF= (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));

	// Histogram equalization steps: 
	// 1. Create the histogram for the input grayscale image.
	CalculateHistogram(imageIn, width, height, histogram);
	// 2. Calculate the cumulative distribution histogram.
	CalculateCDF(histogram, CDF);
	// 3. Calculate the new gray-level values through the general histogram equalization formula 
	//    and assign new pixel values
	Equalize(imageIn, imageOut, width, height, CDF);

    // write output image:
    stbi_write_png("out.png", width, height, DESIRED_NCHANNELS, imageOut, width * DESIRED_NCHANNELS);
    stbi_write_jpg("out.jpg", width, height, DESIRED_NCHANNELS, imageOut, 100);

	//Free memory
	free(imageIn);
    free(imageOut);
	free(histogram);
    free(CDF);

	return 0;
}



