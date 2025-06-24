#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "lenet5_model.h"
#include "lenet5_weights.c"

// Activaton Functions
float relu(float x) {
    if(x > 0) {
        return x;
    }else {
        return 0;
    }
}

void softmax(float* input, int size) {
    float max_val = -FLT_MAX;

    for (int i = 0; i < size; ++i) {
         if (input[i] > max_val) 
             max_val = input[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        input[i] = expf(input[i] - max_val);
        sum_exp += input[i];
    }

    for (int i = 0; i < size; ++i)
        input[i] /= sum_exp;
}

// Layer Implementation

void convolutional_layer(const float* input, float* output, const float* weights, const float* biases, int in_depth, int in_h, int in_w, int out_depth, int out_h, int out_w, int kernel_size) {
    for (int od = 0; od < out_depth; ++od) {
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                float sum = 0.0f;
                for (int id = 0; id < in_depth; ++id) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int iy = oy + ky, ix = ox + kx;
                            sum += input[(id * in_h * in_w) + (iy * in_w) + ix] * weights[(od * in_depth * kernel_size * kernel_size) + (id * kernel_size * kernel_size) + (ky * kernel_size) + kx];
                        }
                    }
                }
                output[(od * out_h * out_w) + (oy * out_w) + ox] = relu(sum + biases[od]);
            }
        }
    }
}


void average_pooling_layer(const float* input, float* output, int depth, int in_h, int in_w, int out_h, int out_w, int pool_size, int stride) {
    float pool_area = (float)(pool_size * pool_size);
    for (int d = 0; d < depth; ++d) {
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                float sum = 0.0f;
                for (int py = 0; py < pool_size; ++py) {
                    for (int px = 0; px < pool_size; ++px) {
                        int iy = oy * stride + py, ix = ox * stride + px;
                        sum += input[(d * in_h * in_w) + (iy * in_w) + ix];
                    }
                }
                output[(d * out_h * out_w) + (oy * out_w) + ox] = sum / pool_area;
            }
        }
    }
}

void flatten_layer(const float* input, float* output, int depth, int height, int width) {
    memcpy(output, input, depth * height * width * sizeof(float));
}

void fully_connected_layer(const float* input, float* output, const float* weights, const float* biases, int in_features, int out_features, int apply_relu) {
    for (int i = 0; i < out_features; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < in_features; ++j) {
            sum += input[j] * weights[i * in_features + j];
        }
        sum += biases[i];
        output[i] = apply_relu ? relu(sum) : sum;
    }
}

void preprocess_image(const unsigned char* input_image, float* output_padded) {
    memset(output_padded, 0, INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
    for (int y = 0; y < MNIST_HEIGHT; ++y) {
        for (int x = 0; x < MNIST_WIDTH; ++x) {
            int padded_y = y + 2;
            int padded_x = x + 2;
            float pixel_val = input_image[y * MNIST_WIDTH + x] / 255.0f;
            output_padded[padded_y * INPUT_WIDTH + padded_x] = (pixel_val - NORM_MEAN) / NORM_STD_DEV;
        }
    }
}

// Main Inference Function

int predict(const unsigned char* image) {
    preprocess_image(image, (float*)input_padded);

    convolutional_layer((float*)input_padded, (float*)c1_output, (float*)c1_weights, c1_biases, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, C1_FILTERS, C1_OUTPUT_HEIGHT, C1_OUTPUT_WIDTH, C1_KERNEL_SIZE);
    average_pooling_layer((float*)c1_output, (float*)s2_output, C1_FILTERS, C1_OUTPUT_HEIGHT, C1_OUTPUT_WIDTH, S2_OUTPUT_HEIGHT, S2_OUTPUT_WIDTH, S2_POOL_SIZE, S2_STRIDE);
    convolutional_layer((float*)s2_output, (float*)c3_output, (float*)c3_weights, c3_biases, C1_FILTERS, S2_OUTPUT_HEIGHT, S2_OUTPUT_WIDTH, C3_FILTERS, C3_OUTPUT_HEIGHT, C3_OUTPUT_WIDTH, C3_KERNEL_SIZE);
    average_pooling_layer((float*)c3_output, (float*)s4_output, C3_FILTERS, C3_OUTPUT_HEIGHT, C3_OUTPUT_WIDTH, S4_OUTPUT_HEIGHT, S4_OUTPUT_WIDTH, S4_POOL_SIZE, S4_STRIDE);
    
    flatten_layer((float*)s4_output, flattened_output, C3_FILTERS, S4_OUTPUT_HEIGHT, S4_OUTPUT_WIDTH);

    fully_connected_layer(flattened_output, f5_output, (float*)f5_weights, f5_biases, FLATTEN_SIZE, F5_OUTPUTS, 1);
    fully_connected_layer(f5_output, f6_output, (float*)f6_weights, f6_biases, F5_OUTPUTS, F6_OUTPUTS, 1);
    fully_connected_layer(f6_output, final_output, (float*)output_weights, output_biases, F6_OUTPUTS, OUTPUT_SIZE, 0);

    softmax(final_output, OUTPUT_SIZE);

    int prediction = -1;
    float max_prob = -1.0f;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        if (final_output[i] > max_prob) {
            max_prob = final_output[i];
            prediction = i;
        }
    }
    return prediction;
}

int main() {
    unsigned char sample_image[MNIST_WIDTH * MNIST_HEIGHT] = {
        // Hardcoded MNIST '2' image data...
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,18,172,253,253,253,253,195,87,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,224,252,253,252,252,252,252,233,155,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,196,252,252,252,252,252,252,252,252,252,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,153,252,252,252,252,240,210,252,252,252,122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,187,253,253,253,139,39,169,252,252,122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,253,253,210,143,221,252,252,210,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,253,253,253,253,253,252,238,143,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,166,253,253,253,253,252,194,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,209,252,252,252,213,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,252,252,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,216,246,173,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,252,252,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,203,252,252,252,114,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,220,252,252,252,252,252,219,48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,151,252,252,252,252,252,252,252,220,48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,185,252,252,252,252,252,252,252,252,252,122,0,0,0,0,0,0,0,0,0,0,0,0,0,118,246,252,252,200,159,159,159,200,252,252,122,0,0,0,0,0,0,0,0,0,0,0,0,0,126,253,252,173,87,22,0,22,87,173,252,229,3,0,0,0,0,0,0,0,0,0,0,0,0,0,31,201,252,219,41,0,0,0,41,219,252,122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };

    printf("Starting LeNet-5 inference...\n");
    int prediction = predict(sample_image);
    printf("Model prediction: %d\n", prediction);
    
    printf("\nProbabilities for each class:\n");
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        printf("  Class %d: %.4f%%\n", i, final_output[i] * 100.0f);
    }

    return 0;
}