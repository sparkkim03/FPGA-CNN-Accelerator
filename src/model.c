#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

#include "conv.h"
#include "pooling.h" 
#include "dense.h"
#include "util/util.h"

// Represeents the single channel input image
float input_one[INPUT_SIZE_ONE][INPUT_SIZE_ONE];
// Represents the six channel output of first convolution layer
float output_one[NUM_FILTER_ONE][OUTPUT_SIZE_ONE][OUTPUT_SIZE_ONE];
// Reppresents the six channel input to the second convolution layer
float input_two[CHANNEL_TWO][INPUT_SIZE_TWO][INPUT_SIZE_TWO];
// Represents the 16 channel output of the second convolution layer
float output_two[NUM_FILTER_TWO][OUTPUT_SIZE_TWO][OUTPUT_SIZE_TWO];
float pool_one_output[POOL_INPUT_CHANNEL_ONE][POOL_OUTPUT_SIZE_ONE][POOL_OUTPUT_SIZE_ONE];
float pool_two_output[POOL_INPUT_CHANNEL_TWO][POOL_OUTPUT_SIZE_TWO][POOL_OUTPUT_SIZE_TWO];

float dense_one_input[FC_LAYER_SIZE_ONE]; 
// Outputs of the dense layers
float dense_one_output[FC_LAYER_SIZE_TWO];
float dense_two_output[FC_LAYER_SIZE_THREE];
float final_output[FC_LAYER_SIZE_FOUR]; 

Filter filter_array_one[NUM_FILTER_ONE]; // Six filters in the first conv layer
Filter filter_array_two[NUM_FILTER_TWO * CHANNEL_TWO]; // 16 filters in the second conv layer

void softmax(float* input, int size) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        // Subtract max_val for numerical stability
        sum_exp += expf(input[i] - max_val);
    }

    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max_val) / sum_exp;
    }
}

void print_output_vector(float* vector, int size, const char* name) {
    printf("  %s:\n    [", name);
    for (int i = 0; i < size; i++) {
        printf("%8.6f%s", vector[i], (i == size - 1) ? "" : ", ");
    }
    printf("]\n\n");
}

int main() {
    char * image_to_classify = "src/data/digit_7_index_70.txt";

    load_image(image_to_classify, input_one);

    // Initialize the convlution kernels with correct weights
    for (int i = 0; i < NUM_FILTER_ONE; i++) initFilter_One(&filter_array_one[i], i);
    for (int i = 0; i < (NUM_FILTER_TWO * CHANNEL_TWO); i++) initFilter_Two(&filter_array_two[i], i);

    // First convolution layer (28*28 -> 24*24)
    convolve_one(input_one, output_one, filter_array_one);
    // Apply reLU
    reLU((float *)output_one, NUM_FILTER_ONE * OUTPUT_SIZE_ONE * OUTPUT_SIZE_ONE);
    // Max Pooling layer 1 (24 * 24 -> 12*12)
    maxpool_one(output_one, pool_one_output);
     
    // Second convolution layer (12*12 -> 8*8)
    convolve_two(pool_one_output, output_two, filter_array_two);
    // Apply reLU
    reLU((float *)output_two, NUM_FILTER_TWO * OUTPUT_SIZE_TWO * OUTPUT_SIZE_TWO);
    // Max Pooling layer 2 (8*8 -> 4*4)
    maxpool_two(output_two, pool_two_output);

    // Flatten the tensor
    for(int c = 0; c < POOL_INPUT_CHANNEL_TWO; c++) {
        for(int i = 0; i < POOL_OUTPUT_SIZE_TWO; i++) {
            for(int j = 0; j < POOL_OUTPUT_SIZE_TWO; j++) {
                int flat_idx = c * (POOL_OUTPUT_SIZE_TWO * POOL_OUTPUT_SIZE_TWO) + i * POOL_OUTPUT_SIZE_TWO + j;
                dense_one_input[flat_idx] = pool_two_output[c][i][j];
            }
        }
    }

    // First FC layer
    dense_one(dense_one_input, dense_one_output);
    reLU(dense_one_output, FC_LAYER_SIZE_TWO);
    // Second FC layer
    dense_two(dense_one_output, dense_two_output);
    reLU(dense_two_output, FC_LAYER_SIZE_THREE);

    // Third FC layer (output)
    dense_three(dense_two_output, final_output);

    // Get the argMax of probability
    int predicted_num = argmax(final_output, 10);

    printf("Final prediction for this image is %d\n", predicted_num);

    return 0;
}

