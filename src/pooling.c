#include <float.h>

#include "pooling.h"

void maxpool_one(float input[POOL_INPUT_CHANNEL_ONE][POOL_INPUT_SIZE_ONE][POOL_INPUT_SIZE_ONE], 
    float output[POOL_INPUT_CHANNEL_ONE][POOL_OUTPUT_SIZE_ONE][POOL_OUTPUT_SIZE_ONE]) {

    for(int c = 0; c < POOL_INPUT_CHANNEL_ONE; c++) {
        for(int o_r = 0; o_r < POOL_OUTPUT_SIZE_ONE; o_r++) {
            for(int o_c = 0; o_c < POOL_OUTPUT_SIZE_ONE; o_c++) {
                // Our stride is 2
                int i_r = o_r * POOL_STRIDE; // starting point of the row (offset)
                int i_c = o_c * POOL_STRIDE; // starting point of the column (offset)

                float max = -FLT_MAX;

                for(int i = 0; i < POOL_SIZE; i++) {
                    for(int j = 0; j < POOL_SIZE; j++) {
                        float curr = input[c][i_r + i][i_c + j];

                        if(curr > max) {
                            max = curr;
                        }
                    }
                }

                output[c][o_r][o_c] = max;
            }
        }
    }
}

void maxpool_two(float input[POOL_INPUT_CHANNEL_TWO][POOL_INPUT_SIZE_TWO][POOL_INPUT_SIZE_TWO], 
    float output[POOL_INPUT_CHANNEL_TWO][POOL_OUTPUT_SIZE_TWO][POOL_OUTPUT_SIZE_TWO]) {

    for(int c = 0; c < POOL_INPUT_CHANNEL_TWO; c++) {
        for(int o_r = 0; o_r < POOL_OUTPUT_SIZE_TWO; o_r++) {
            for(int o_c = 0; o_c < POOL_OUTPUT_SIZE_TWO; o_c++) {
                // Our stride is 2
                int i_r = o_r * POOL_STRIDE; // starting point of `the row (offset)
                int i_c = o_c * POOL_STRIDE; // starting point of the column (offset)

                float max = -FLT_MAX;

                for(int i = 0; i < POOL_SIZE; i++) {
                    for(int j = 0; j < POOL_SIZE; j++) {
                        float curr = input[c][i_r + i][i_c + j];

                        if(curr > max) {
                            max = curr;
                        }
                    }
                }
                output[c][o_r][o_c] = max;
            }
        }
    }
}