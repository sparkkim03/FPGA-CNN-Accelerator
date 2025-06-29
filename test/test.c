#include <stdio.h>
#include <float.h>
#include <stdlib.h>

#include "../src/conv.h"
#include "../src/pooling.h" 
#include "../src/dense.h"

// --- CORRECTED: Declare output arrays for the pooling layers as multi-dimensional arrays ---
float pool_one_output[POOL_INPUT_CHANNEL_ONE][POOL_OUTPUT_SIZE_ONE][POOL_OUTPUT_SIZE_ONE];
float pool_two_output[POOL_INPUT_CHANNEL_TWO][POOL_OUTPUT_SIZE_TWO][POOL_OUTPUT_SIZE_TWO];

float dense_one_input[FC_LAYER_SIZE_ONE]; 
// Outputs of the dense layers
float dense_one_output[FC_LAYER_SIZE_TWO];
float dense_two_output[FC_LAYER_SIZE_THREE];
float final_output[FC_LAYER_SIZE_FOUR]; // Before softmax

// Helper to print a feature map corner
void print_feature_map_corner(float* map, int map_size, int map_idx, const char* map_name);
// Helper to print the final output vector
void print_output_vector(float* vector, int size, const char* name);
// Softmax function to compute final probabilities
void softmax(float* input, int size);

// --- Helper Function Implementations ---
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

void print_feature_map_corner(float* map, int map_size, int map_idx, const char* map_name) {
    printf("  %s %d (Top-Left Corner):\n", map_name, map_idx);
    int print_size = (map_size < 5) ? map_size : 5;
    for (int i = 0; i < print_size; i++) {
        printf("    ");
        for (int j = 0; j < print_size; j++) {
            int idx = i * map_size + j;
            printf("%+10.6f ", map[idx]); 
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    printf("--- Initializing Filters ---\n");
    for (int i = 0; i < NUM_FILTER_ONE; i++) initFilter_One(&filter_array_one[i], i);
    for (int i = 0; i < (NUM_FILTER_TWO * CHANNEL_TWO); i++) initFilter_Two(&filter_array_two[i], i);
    printf("Initialization complete.\n\n");

    printf("--- Preparing Test Input ---\n");
    for (int i = 0; i < INPUT_SIZE_ONE; i++) {
        for (int j = 0; j < INPUT_SIZE_ONE; j++) {
            input_one[i][j] = (float)(i + j) / 100.0f;
        }
    }
    printf("Input prepared.\n\n");

    // --- CONV1 -> RELU -> POOL1 Pipeline ---
    printf("--- Running CONV1 ---\n");
    convolve_one(input_one, NULL);
    printf("CONV1 complete. Printing results (pre-activation)...\n\n");
    print_feature_map_corner((float*)output_one[0], OUTPUT_SIZE_ONE, 0, "Feature Map (Pre-ReLU)");
    
    printf("--- Applying ReLU to CONV1 Output ---\n");
    reLU((float*)output_one, NUM_FILTER_ONE * OUTPUT_SIZE_ONE * OUTPUT_SIZE_ONE);
    printf("ReLU complete. Printing results...\n\n");
    print_feature_map_corner((float*)output_one[0], OUTPUT_SIZE_ONE, 0, "Feature Map (Post-ReLU)");

    printf("--- Running POOL1 ---\n");
    maxpool_one(output_one, pool_one_output);
    printf("POOL1 complete. Printing results...\n\n");
    print_feature_map_corner((float*)pool_one_output[0], POOL_OUTPUT_SIZE_ONE, 0, "Pooled Map");
    
    // --- CONV2 -> RELU -> POOL2 Pipeline ---
    printf("--- Running CONV2 ---\n");
    convolve_two(pool_one_output, NULL);
    printf("CONV2 complete. Printing results (pre-activation)...\n\n");
    print_feature_map_corner((float*)output_two[0], OUTPUT_SIZE_TWO, 0, "Feature Map (Pre-ReLU)");
    
    printf("--- Applying ReLU to CONV2 Output ---\n");
    reLU((float*)output_two, NUM_FILTER_TWO * OUTPUT_SIZE_TWO * OUTPUT_SIZE_TWO);
    printf("ReLU complete. Printing results...\n\n");
    print_feature_map_corner((float*)output_two[0], OUTPUT_SIZE_TWO, 0, "Feature Map (Post-ReLU)");
    
    printf("--- Running POOL2 ---\n");
    maxpool_two(output_two, pool_two_output);
    printf("POOL2 complete. Printing results...\n\n");
    print_feature_map_corner((float*)pool_two_output[0], POOL_OUTPUT_SIZE_TWO, 0, "Pooled Map");

    // --- Flatten and Dense Layers ---
    printf("--- Flattening Output for Dense Layers ---\n");
    for(int c = 0; c < POOL_INPUT_CHANNEL_TWO; c++) {
        for(int i = 0; i < POOL_OUTPUT_SIZE_TWO; i++) {
            for(int j = 0; j < POOL_OUTPUT_SIZE_TWO; j++) {
                int flat_idx = c * (POOL_OUTPUT_SIZE_TWO * POOL_OUTPUT_SIZE_TWO) + i * POOL_OUTPUT_SIZE_TWO + j;
                dense_one_input[flat_idx] = pool_two_output[c][i][j];
            }
        }
    }
    printf("Flattening complete.\n\n");
    print_output_vector(dense_one_input, FC_LAYER_SIZE_ONE, "Logits (INITIAL)");

    printf("--- Running DENSE1 -> RELU ---\n");
    dense_one(dense_one_input, dense_one_output);
    reLU(dense_one_output, FC_LAYER_SIZE_TWO);
    print_output_vector(dense_one_output, FC_LAYER_SIZE_TWO, "Logits (FC1)");
    
    printf("--- Running DENSE2 -> RELU ---\n");
    dense_two(dense_one_output, dense_two_output);
    reLU(dense_two_output, FC_LAYER_SIZE_THREE);
    print_output_vector(dense_two_output, FC_LAYER_SIZE_THREE, "Logits (FC2)");

    printf("--- Running DENSE3 (Output Layer) ---\n");
    dense_three(dense_two_output, final_output);
    print_output_vector(final_output, FC_LAYER_SIZE_FOUR, "Logits (Output Before Softmax)");

    printf("--- Applying Softmax ---\n");
    softmax(final_output, FC_LAYER_SIZE_FOUR);
    print_output_vector(final_output, FC_LAYER_SIZE_FOUR, "Final Probabilities");

    return 0;
}