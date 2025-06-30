#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

void load_image(const char * filename, float img_array[IMG_HEIGHT][IMG_WIDTH]) {
    FILE * fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("Error opening image file");
        exit(1); 
    }

    int pixel_value;
    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            // Read one integer value from the file.
            // fscanf automatically handles spaces and newlines.
            if (fscanf(fp, "%d", &pixel_value) != 1) {
                fprintf(stderr, "Error: Failed to read pixel value at (%d, %d).\n", i, j);
                fclose(fp);
                exit(1);
            }

            printf("Pixel Value %d\n", pixel_value);
            // Normalize the pixel value from [0, 255] to [0.0, 1.0] and store it.
            img_array[i][j] = (float)pixel_value / 255.0f;
        }
    }
    fclose(fp);
}

int argmax(float * input, int size) {
    if (size <= 0) {
        return -1; // Handle invalid size
    }

    int max_idx = 0;
    float max_val = input[0];

    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }

    return max_idx;
}