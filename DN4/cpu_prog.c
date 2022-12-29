#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define COLOR_CHANNELS 4
#define BLOCK_SIZE 16

#ifndef MAX
    #define MAX(a, b) (a > b ? a : b)
#endif

#ifndef MIN
    #define MIN(a, b) (a < b ? a : b)
#endif

#define OMP

#include "stb_image.h"
#include "stb_image_write.h"

/**
 * @brief Simple program to sharpen image on the CPU
 * 
 * @param OMP - use parallel programing
 * 
 * Run with:
 * gcc cpu_prog.c -lm -fopenmp -o cpu_prog.out
 * time ./cpu_prog.out helmet_in.png helmet_out.png
 */

unsigned char getIntensity(const unsigned char *image, int row, int col,
                           int channel, int height, int width, int cpp)
{
    if (col < 0 || col >= width)
        return 0;
    if (row < 0 || row >= height)
        return 0;
    return image[(row * width + col) * cpp + channel];
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("USAGE: prog input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    char szImage_out_name[255];

    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);
    cpp = COLOR_CHANNELS;

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    if (imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }

    printf("Loaded image %s of size %dx%d.\n", szImage_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *imageOut = (unsigned char *)malloc(datasize);

    #ifdef OMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < cpp; c++)
            {
                unsigned char px01 = getIntensity(imageIn, y - 1, x, c, height, width, cpp);
                unsigned char px10 = getIntensity(imageIn, y, x - 1, c, height, width, cpp);
                unsigned char px11 = getIntensity(imageIn, y, x, c, height, width, cpp);
                unsigned char px12 = getIntensity(imageIn, y, x + 1, c, height, width, cpp);
                unsigned char px21 = getIntensity(imageIn, y + 1, x, c, height, width, cpp);

                short pxOut = (5 * px11 - px01 - px10 - px12 - px21);
                pxOut = MIN(pxOut, 255);
                pxOut = MAX(pxOut, 0);
                imageOut[(y * width + x) * cpp + c] = (unsigned char)pxOut;
            }
        }
    }

    // Retrieve output file type
    char szImage_out_name_temp[255];
    strncpy(szImage_out_name_temp, szImage_out_name, 255);
    char *token = strtok(szImage_out_name_temp, ".");
    char *FileType = NULL;
    while (token != NULL)
    {
        FileType = token;
        token = strtok(NULL, ".");
    }
    // Write output image to file
    if (!strcmp(FileType, "png"))
        stbi_write_png(szImage_out_name, width, height, cpp, imageOut, width * cpp);
    else if (!strcmp(FileType, "jpg"))
        stbi_write_jpg(szImage_out_name, width, height, cpp, imageOut, 100);
    else if (!strcmp(FileType, "bmp"))
        stbi_write_bmp(szImage_out_name, width, height, cpp, imageOut);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", FileType);
    // Release host memory
    free(imageIn);
    free(imageOut);

    return 0;
}