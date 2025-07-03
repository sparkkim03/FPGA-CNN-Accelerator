#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <xtimer_config.h>

#include "conv.h"
#include "pooling.h" 
#include "dense.h"
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

float input_one[28][28] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 228, 211, 89, 0, 0, 0, 0, 0, 0, 51, 121, 121, 226, 253, 232, 44, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 61, 235, 249, 240, 240, 240, 240, 240, 241, 245, 252, 252, 252, 252, 252, 93, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 64, 183, 252, 252, 252, 252, 252, 253, 252, 252, 252, 252, 252, 252, 93, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 39, 39, 53, 95, 39, 39, 39, 39, 219, 252, 252, 197, 63, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 244, 252, 209, 17, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 122, 252, 252, 198, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 244, 252, 252, 177, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 241, 252, 252, 126, 24, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 197, 252, 252, 248, 50, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 196, 253, 252, 252, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 197, 253, 255, 253, 159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 123, 252, 252, 253, 252, 103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 252, 252, 252, 253, 37, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 198, 252, 252, 209, 110, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 176, 252, 252, 248, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 17, 107, 252, 252, 252, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 171, 252, 252, 252, 233, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 10, 217, 252, 252, 252, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 89, 250, 252, 252, 221, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 213, 217, 217, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};


int main() {
    init_platform();

    // Measuring the time
    XTime tStart, tEnd;

    XTime_GetTime(&tStart);    

    char * image_to_classify = "C:/Users/stefano/FPGA_CNN_Accelerator/vitis-workspace/digit_classifier/src/img.txt";

    //load_image(image_to_classify, input_one);

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

    XTime_GetTime(&tEnd);

    printf("Final prediction for this image is %d\n", predicted_num);

    printf("Prediction took %f ms\n", ((double)(tEnd - tStart)/COUNTS_PER_SECOND)*1000.0);

    cleanup_platform();
    return 0;
}

