#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "conv.h"
#include "pooling.h"

void maxpool(const float* input, int input_channels, int input_size,
             float* output, int output_size) {
    
    // Iterate over each channel of the input feature map
    for (int c = 0; c < input_channels; c++) {
        // Iterate over each pixel of the output feature map
        for (int o_r = 0; o_r < output_size; o_r++) {
            for (int o_c = 0; o_c < output_size; o_c++) {
                
                // Calculate the starting position of the pooling window in the input array
                int i_r = o_r * POOL_STRIDE;
                int i_c = o_c * POOL_STRIDE;
                
                float max = -FLT_MAX;
                
                // Iterate over the 2x2 pooling window
                for (int i = 0; i < POOL_SIZE; i++) {
                    for (int j = 0; j < POOL_SIZE; j++) {
                        // Use pointer arithmetic to access the correct input element
                        int input_idx = c * input_size * input_size + (i_r + i) * input_size + (i_c + j);
                        float curr = input[input_idx];
                        
                        if (curr > max) {
                            max = curr;
                        }
                    }
                }
                
                // Use pointer arithmetic to access the correct output element
                int output_idx = c * output_size * output_size + o_r * output_size + o_c;
                output[output_idx] = max;
            }
        }
    }
}