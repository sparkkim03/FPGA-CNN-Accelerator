#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "conv.h"
#include "lenet_weights.h"

void initFilter_One(Filter * filter, int filterNth) {
    for(int i = 0; i < FILTER_SIZE; i++) {
        for(int j = 0; j < FILTER_SIZE; j++) {
            // i * FILTER_SIZE + j
            int weightIDX = (filterNth * 25) + (i * 5) + j;
            filter->weights[i][j] = conv1_weight[weightIDX];
        }
    }    
    filter->bias = conv1_bias[filterNth];
}

void initFilter_Two(Filter * filter, int filterNth) {
    for(int i = 0; i < FILTER_SIZE; i++) {
        for(int j = 0; j < FILTER_SIZE; j++) {
            // i * FILTER_SIZE + j
            int weightIDX = (filterNth * 25) + (i * 5) + j;
            filter->weights[i][j] = conv2_weight[weightIDX];
        }
    }    
    filter->bias = conv2_bias[filterNth / 6];
}

void convolve_one(float input[INPUT_SIZE_ONE][INPUT_SIZE_ONE], 
    float output[NUM_FILTER_ONE][OUTPUT_SIZE_ONE][OUTPUT_SIZE_ONE],
    Filter filters[NUM_FILTER_ONE]
) {

    // Iterate over each of the 6 filters for the first layer.
    for (int f_idx = 0; f_idx < NUM_FILTER_ONE; f_idx++) {
        Filter* filter = &filters[f_idx];

        // Iterate over each pixel of the output feature map.
        for (int i = 0; i < OUTPUT_SIZE_ONE; i++) {
            for (int j = 0; j < OUTPUT_SIZE_ONE; j++) {
                float sum = 0.0;
                // Apply the 5x5 filter kernel.
                for (int k_i = 0; k_i < FILTER_SIZE; k_i++) {
                    for (int k_j = 0; k_j < FILTER_SIZE; k_j++) {
                        // Because input is 2D, we can index it directly.
                        sum += input[i + k_i][j + k_j] * filter->weights[k_i][k_j];
                    }
                }
                // Add the bias once per output pixel.
                sum += filter->bias;
                output[f_idx][i][j] = sum;
            }
        }
    }
}

void convolve_two(float input[CHANNEL_TWO][INPUT_SIZE_TWO][INPUT_SIZE_TWO], 
    float output[NUM_FILTER_TWO][OUTPUT_SIZE_TWO][OUTPUT_SIZE_TWO],
    Filter filters[NUM_FILTER_TWO * CHANNEL_TWO]
) {

    // Iterate over each of the 16 output feature maps.
    for (int f_idx = 0; f_idx < NUM_FILTER_TWO; f_idx++) {
        // Iterate over each pixel of the output feature map.
        for (int i = 0; i < OUTPUT_SIZE_TWO; i++) {
            for (int j = 0; j < OUTPUT_SIZE_TWO; j++) {
                float total_sum = 0.0;
                // To calculate one output pixel, we must sum the convolutions over all 6 input channels.
                for (int c_idx = 0; c_idx < CHANNEL_TWO; c_idx++) {
                    // Calculate index for the correct filter for the current output map and input channel.
                    int filter_index = f_idx * CHANNEL_TWO + c_idx;
                    Filter* filter = &filters[filter_index]; // Assumes corrected, larger array
                    
                    float channel_sum = 0.0;
                    // Apply the 5x5 filter kernel to the current input channel.
                    for (int k_i = 0; k_i < FILTER_SIZE; k_i++) {
                        for (int k_j = 0; k_j < FILTER_SIZE; k_j++) {
                            channel_sum += input[c_idx][i + k_i][j + k_j] * filter->weights[k_i][k_j];
                        }
                    }
                    total_sum += channel_sum;
                }
                int first_filter_index = f_idx * CHANNEL_TWO;
                total_sum += filters[first_filter_index].bias;
                output[f_idx][i][j] = total_sum;
            }
        }
    }
}

void reLU(float * input, int size) {
    for(int i = 0; i < size; i++) {
        if(input[i] < 0) {
            input[i] = 0;
        }
    }
}