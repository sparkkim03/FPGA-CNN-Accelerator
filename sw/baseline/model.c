#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <xtimer_config.h>

#include "conv.h"
#include "pooling.h" 
#include "dense.h"
#include "lenet_weights.h"
#include "util/util.h"

#include "platform.h"
#include "xil_printf.h"
#include "xil_io.h"
#include "xiltimer.h"

// Represeents the single channel input image
//float input_one[INPUT_SIZE_ONE][INPUT_SIZE_ONE];
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

float input_one[28][28];


int main() {
    init_platform();

    // Measuring the time
    XTime tStart, tEnd;

    XTime_GetTime(&tStart);    

    char * image_to_classify = "C:/Users/stefano/FPGA_CNN_Accelerator/vitis-workspace/digit_classifier/src/img.txt";

    //load_image(image_to_classify, input_one);

    // Initialize the convlution kernels with correct weights

    for(int i = 0; i < NUM_FILTER_ONE; i++) initFilter(&filter_array_one[i], i, 0);
    for (int i = 0; i < (NUM_FILTER_TWO * CHANNEL_TWO); i++) initFilter_Two(&filter_array_two[i], i, 1);

    // First convolution layer (28*28 -> 24*24)
    convolve(input_one, CHANNEL_ONE, INPUT_SIZE_ONE, output_two, NUM_FILTER_ONE, OUTPUT_SIZE_ONE, filter_array_one);
    // Apply reLU
    reLU((float *)output_one, NUM_FILTER_ONE * OUTPUT_SIZE_ONE * OUTPUT_SIZE_ONE);
    // Max Pooling layer 1 (24 * 24 -> 12*12)
    maxpool(output_one, POOL_INPUT_CHANNEL_ONE, POOL_INPUT_SIZE_ONE, pool_one_output, POOL_OUTPUT_SIZE_ONE);
    // Second convolution layer (12*12 -> 8*8)
    convolve(pool_one_output, CHANNEL_TWO, INPUT_SIZE_TWO, output_two, NUM_FILTER_TWO, OUTPUT_SIZE_TWO, filter_array_two);
    // Apply reLU
    reLU((float *)output_two, NUM_FILTER_TWO * OUTPUT_SIZE_TWO * OUTPUT_SIZE_TWO);
    // Max Pooling layer 2 (8*8 -> 4*4)
    maxpool(output_two, POOL_INPUT_CHANNEL_TWO, POOL_INPUT_SIZE_TWO, pool_two_output, POOL_OUTPUT_SIZE_TWO);

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
    dense(dense_one_input, FC_LAYER_SIZE_ONE, dense_one_output, FC_LAYER_SIZE_TWO, fc1_weight, fc1_bias);
    reLU(dense_one_output, FC_LAYER_SIZE_TWO);
    // Second FC layer
    dense(dense_one_output, FC_LAYER_SIZE_TWO, dense_two_output, FC_LAYER_SIZE_THREE, fc2_weight, fc2_bias);
    reLU(dense_two_output, FC_LAYER_SIZE_THREE);

    // Third FC layer (output)
    dense(dense_two_output, FC_LAYER_SIZE_THREE, final_output, FC_LAYER_SIZE_FOUR, fc3_weight, fc3_bias);

    // Get the argMax of probability
    int predicted_num = argmax(final_output, 10);

    XTime_GetTime(&tEnd);

    printf("Final prediction for this image is %d\n", predicted_num);

    printf("Prediction took %f ms\n", ((double)(tEnd - tStart)/COUNTS_PER_SECOND)*1000.0);

    cleanup_platform();
    return 0;
}

