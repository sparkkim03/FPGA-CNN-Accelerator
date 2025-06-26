#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PADDED_WIDTH 32
#define PADDED_HEIGHT 32
#define ORIGINAL_WIDTH 28
#define ORIGINAL_HEIGHT 28

/**
 * @brief Reads a PGM image file and loads it into a padded float array.
 * * @param filename The path to the .pgm file.
 * @param output_array The destination float array (should be PADDED_HEIGHT * PADDED_WIDTH).
 * @return 0 on success, -1 on failure.
 */
int read_pgm_to_padded_array(const char* filename, float* output_array) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return -1;
    }

    // --- Parse PGM Header ---
    char magic[3];
    int width, height, max_val;

    // Read magic number (should be "P2")
    fscanf(fp, "%2s", magic);
    if (strcmp(magic, "P2") != 0) {
        fprintf(stderr, "Error: Not a valid P2 PGM file.\n");
        fclose(fp);
        return -1;
    }

    // Read width, height, and max pixel value
    if (fscanf(fp, "%d %d %d", &width, &height, &max_val) != 3) {
        fprintf(stderr, "Error: Could not read PGM header info.\n");
        fclose(fp);
        return -1;
    }

    if (width != ORIGINAL_WIDTH || height != ORIGINAL_HEIGHT) {
        fprintf(stderr, "Warning: Image dimensions (%dx%d) are not the expected 28x28.\n", width, height);
    }
    
    printf("Reading PGM file: %s (%dx%d, max_val: %d)\n", filename, width, height, max_val);

    // --- Read Pixel Data and Populate Array ---
    int padding_ho = (PADDED_HEIGHT - ORIGINAL_HEIGHT) / 2; // horizontal offset
    int padding_wo = (PADDED_WIDTH - ORIGINAL_WIDTH) / 2;   // vertical offset

    // Initialize the entire padded array to 0.0f
    for (int i = 0; i < PADDED_HEIGHT * PADDED_WIDTH; ++i) {
        output_array[i] = 0.0f;
    }

    // Read pixel values and place them in the center of the padded array
    for (int h = 0; h < ORIGINAL_HEIGHT; ++h) {
        for (int w = 0; w < ORIGINAL_WIDTH; ++w) {
            int pixel;
            if (fscanf(fp, "%d", &pixel) != 1) {
                fprintf(stderr, "Error: Ran out of data while reading pixels.\n");
                fclose(fp);
                return -1;
            }
            
            // Calculate the index in the larger padded array
            int padded_idx = (h + padding_ho) * PADDED_WIDTH + (w + padding_wo);
            
            // Normalize the pixel value from [0, 255] to [0.0, 1.0]
            output_array[padded_idx] = (float)pixel / max_val;
        }
    }

    fclose(fp);
    return 0;
}

// Example usage
int main() {
    // This array will hold our image data, just like sample_image.h
    float my_image_array[PADDED_HEIGHT * PADDED_WIDTH];

    // Assuming you ran the python script and have this file
    const char* image_to_load = "../model/mnist_pgm_images/mnist_img_0_label_5.pgm";

    if (read_pgm_to_padded_array(image_to_load, my_image_array) == 0) {
        printf("Successfully loaded %s into a float array.\n", image_to_load);

        // You can now pass `my_image_array` to your `convolution_layer` function.
        // For example, to verify, let's print a small section from the center.
        printf("--- Sample from center of padded array ---\n");
        int center_h = PADDED_HEIGHT / 2;
        int center_w = PADDED_WIDTH / 2;
        for (int h_offset = -2; h_offset <= 2; ++h_offset) {
            for (int w_offset = -2; w_offset <=2; ++w_offset) {
                int idx = (center_h + h_offset) * PADDED_WIDTH + (center_w + w_offset);
                printf("%.3f ", my_image_array[idx]);
            }
            printf("\n");
        }
    } else {
        fprintf(stderr, "Failed to load image.\n");
    }

    return 0;
}