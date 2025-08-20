#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "conv.h"
#include "lenet_weights.h"

void initFilter(Filter * filter, int filterNth, int filterNum) {
    for(int i = 0; i < FILTER_SIZE; i++) {
        for(int j = 0; j < FILTER_SIZE; j++) {
            // i * FILTER_SIZE + j
            int weightIDX = (filterNth * 25) + (i * 5) + j;
            if(filterNum == 0) {
                filter->weights[i][j] = conv1_weight[weightIDX];
            }else {
                filter->weights[i][j] = conv2_weight[weightIDX];
            }
        }
    }    
    if(filterNum == 0) {
        filter->bias = conv1_bias[filterNth];
    }else {
        filter->bias = conv2_bias[filterNth / 6];
    }
}

void convolve(const float *input,
              int input_channels,
              int input_size,
              float *output,
              int num_filters,
              int output_size,
              const Filter *filters) {

    // Iterate over each of the output feature maps (filters)
    for (int f_idx = 0; f_idx < num_filters; f_idx++) {
        // Iterate over each pixel of the output feature map
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < output_size; j++) {
                float total_sum = 0.0;
                
                // To calculate one output pixel, sum the convolutions over all input channels
                for (int c_idx = 0; c_idx < input_channels; c_idx++) {
                    // Each output feature map requires a unique filter for each input channel
                    int filter_index = f_idx * input_channels + c_idx;
                    const Filter* filter = &filters[filter_index];
                    
                    float channel_sum = 0.0;
                    // Apply the filter kernel to the current input channel
                    for (int k_i = 0; k_i < FILTER_SIZE; k_i++) {
                        for (int k_j = 0; k_j < FILTER_SIZE; k_j++) {
                            // Calculate the index for the input array using pointer arithmetic
                            int input_idx = c_idx * input_size * input_size + (i + k_i) * input_size + (j + k_j);
                            channel_sum += input[input_idx] * filter->weights[k_i][k_j];
                        }
                    }
                    total_sum += channel_sum;
                }
                
                // The bias is added only once per output pixel, after summing all channels
                int first_filter_index = f_idx * input_channels;
                total_sum += filters[first_filter_index].bias;
                
                // Calculate the index for the output array using pointer arithmetic
                int output_idx = f_idx * output_size * output_size + i * output_size + j;
                output[output_idx] = total_sum;
            }
        }
    }
}