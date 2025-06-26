#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// The pre-trained weights for the LeNet model
#include "lenet_weights.h"

// Only for initial testing purposes
#include "sample_image.h"

#define PADDED_WIDTH 32
#define PADDED_HEIGHT 32
#define ORIGINAL_WIDTH 28
#define ORIGINAL_HEIGHT 28

// Activation function
float tanh_activation(float x) {
    return tanhf(x);
}

void convolution_layer(const float* input, float* output, const float* weights, const float* biases,
                       int in_h, int in_w, int in_d,
                       int out_h, int out_w, int out_d,
                       int k_h, int k_w, int padding) {
    
    printf("Executing Convolution Layer: (%dx%dx%d) -> (%dx%dx%d) with %dx%d kernel\n", 
        in_h, in_w, in_d, out_h, out_w, out_d, k_h, k_w);
                       
    // Matrix multiplication for convolution in C is pain 
    // Loop over each output filter 
    for (int od = 0; od < out_d; ++od) {
        // Loop over the height of the output feature map
        for (int oh = 0; oh < out_h; ++oh) {
            // Loop over the width of the output feature map
            for (int ow = 0; ow < out_w; ++ow) {
                float sum = 0.0f;
                // Loop over the depth (channels) of the input
                for (int id = 0; id < in_d; ++id) {
                    // Loop over the kernel height
                    for (int kh = 0; kh < k_h; ++kh) {
                        // Loop over the kernel width
                        for (int kw = 0; kw < k_w; ++kw) {
                            int ih = oh + kh - padding;
                            int iw = ow + kw - padding;

                            // Check for bounds (valid padding)
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                // Input index
                                int input_idx = id * (in_h * in_w) + ih * in_w + iw;
                                // Weight index
                                int weight_idx = od * (in_d * k_h * k_w) + id * (k_h * k_w) + kh * k_w + kw;
                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }
                // Add bias for the current filter
                sum += biases[od];
                // Output index and activation
                int output_idx = od * (out_h * out_w) + oh * out_w + ow;
                output[output_idx] = tanh_activation(sum);
            }
        }
    }
}

// AVG POOL 2D
void avg_pool_layer(const float* input, float* output,
                    int in_h, int in_w, int in_d, int pool_size) {

    int out_h = in_h / pool_size;
    int out_w = in_w / pool_size;
    printf("Executing Avg Pooling Layer: (%dx%dx%d) -> (%dx%dx%d) with %dx%d pool\n",
           in_h, in_w, in_d, out_h, out_w, in_d, pool_size, pool_size);

    // Loop over each channel
    for (int d = 0; d < in_d; ++d) {
        // Loop over output height
        for (int oh = 0; oh < out_h; ++oh) {
            // Loop over output width
            for (int ow = 0; ow < out_w; ++ow) {
                float sum = 0.0f;
                // Loop over the pooling window
                for (int ph = 0; ph < pool_size; ++ph) {
                    for (int pw = 0; pw < pool_size; ++pw) {
                        int ih = oh * pool_size + ph;
                        int iw = ow * pool_size + pw;
                        int input_idx = d * (in_h * in_w) + ih * in_w + iw;
                        sum += input[input_idx];
                    }
                }
                int output_idx = d * (out_h * out_w) + oh * out_w + ow;
                output[output_idx] = sum / (pool_size * pool_size);
            }
        }
    }
}

// FC layer
void dense_layer(const float* input, float* output, const float* weights, const float* biases,
                 int in_features, int out_features) {
    printf("Executing Dense Layer: %d -> %d features\n", in_features, out_features);

    for (int out_idx = 0; out_idx < out_features; ++out_idx) {
        float sum = 0.0f;
        for (int in_idx = 0; in_idx < in_features; ++in_idx) {
            sum += input[in_idx] * weights[in_idx * out_features + out_idx];
        }
        sum += biases[out_idx];
        output[out_idx] = tanh_activation(sum);
    }
}


int output_layer_predict(const float* input, const float* weights, const float* biases,
                         int in_features, int out_features) {
    printf("Executing Output Layer: %d -> %d features\n", in_features, out_features);
    float output_scores[out_features];

    // Same logic as a dense layer but without the final activation
    for (int out_idx = 0; out_idx < out_features; ++out_idx) {
        float sum = 0.0f;
        for (int in_idx = 0; in_idx < in_features; ++in_idx) {
            sum += input[in_idx] * weights[in_idx * out_features + out_idx];
        }
        sum += biases[out_idx];
        output_scores[out_idx] = sum;
    }

    // Find the index of the maximum score (this is argmax, which is what we need for inference)
    int max_idx = 0;
    float max_score = output_scores[0];
    printf("\nFinal Scores (Logits):\n");
    for (int i = 0; i < out_features; ++i) {
        printf("  Digit %d: %.4f\n", i, output_scores[i]);
        if (output_scores[i] > max_score) {
            max_score = output_scores[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// Helper function to read a PGM file and convert it to a padded float array
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

int main() {
    printf("--- LeNet-5 Inference in C ---\n");

    float my_image_array[PADDED_HEIGHT * PADDED_WIDTH];

    // Assuming you ran the python script and have this file
    const char* image_to_load = "../model/mnist_pgm_images/mnist_img_4_label_9.pgm";

    if (read_pgm_to_padded_array(image_to_load, my_image_array) == 0) {
        printf("Successfully loaded %s into a float array.\n", image_to_load);
    } else {
        fprintf(stderr, "Failed to load image.\n");
    }

    // --- Allocate Memory for Layer Outputs ---
    // These buffers will hold the output of each layer.

    // C1: Conv Layer 1 (Input: 32x32x1, Output: 28x28x6)
    // Keras 'same' padding on a 32x32 input with a 5x5 kernel results in a 32x32 output.
    float* c1_output = (float*)malloc(32 * 32 * 6 * sizeof(float));

    // S2: Pooling Layer 1 (Input: 32x32x6, Output: 16x16x6)
    float* s2_output = (float*)malloc(16 * 16 * 6 * sizeof(float));

    // C3: Conv Layer 2 (Input: 16x16x6, Output: 12x12x16)
    // Keras 'valid' padding: 16 - 5 + 1 = 12
    float* c3_output = (float*)malloc(12 * 12 * 16 * sizeof(float));

    // S4: Pooling Layer 2 (Input: 12x12x16, Output: 6x6x16)
    float* s4_output = (float*)malloc(6 * 6 * 16 * sizeof(float));
    
    // C5: Dense Layer 1 (Input: 6*6*16=576, Output: 120)
    // For original LeNet: Pool(12x12) -> 6x6. Flatten -> 6*6*16 = 576
    float* c5_output = (float*)malloc(120 * sizeof(float));

    // F6: Dense Layer 2 (Input: 120, Output: 84)
    float* f6_output = (float*)malloc(84 * sizeof(float));

    // --- Execute the Forward Pass ---
    
    // Layer 1: Convolution (C1)
    // Input: 32x32x1, Kernel: 5x5, Filters: 6 -> Output: 28x28x6
    convolution_layer(my_image_array, c1_output, C1_weights, C1_biases,
                      32, 32, 1, 28, 28, 6, 5, 5, 0);

    // Layer 2: Pooling (S2)
    // Input: 28x28x6, Pool: 2x2 -> Output: 14x15x6
    avg_pool_layer(c1_output, s2_output, 28, 28, 6, 2);

    // Layer 3: Convolution (C3)
    // Input: 16x16x6, Kernel: 5x5, Filters: 16, Padding: 0 ('valid') -> Output: 12x12x16
    convolution_layer(s2_output, c3_output, C3_weights, C3_biases,
                      14, 14, 6, 10, 10, 16, 5, 5, 0);
                      
    // Layer 4: Pooling (S4)
    // Input: 12x12x16, Pool: 2x2 -> Output: 6x6x16
    avg_pool_layer(c3_output, s4_output, 10, 10, 16, 2);

    // The output of S4 is flattened before being passed to the dense layers.
    // 5 * 5 * 16 = 400
    float* flattened_input = s4_output;

    // Layer 5: Dense (C5)
    // Input: 400, Output: 120
    dense_layer(flattened_input, c5_output, C5_FC1_weights, C5_FC1_biases, 400, 120);

    // Layer 6: Dense (F6)
    // Input: 120, Output: 84
    dense_layer(c5_output, f6_output, FC2_weights, FC2_biases, 120, 84);

    // Layer 7: Output Layer
    // Input: 84, Output: 10
    int prediction = output_layer_predict(f6_output, Output_weights, Output_biases, 84, 10);

    printf("\n----------------------------------\n");
    printf("Final Prediction for the sample image is: %d\n", prediction);
    printf("----------------------------------\n");

    // --- Free allocated memory ---
    free(c1_output);
    free(s2_output);
    free(c3_output);
    free(s4_output);
    free(c5_output);
    free(f6_output);

    return 0;
}