#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "dense.h"
#include "lenet_weights.h"

void dense_one(float * input, float * output) {
    for(int o = 0; o < FC_LAYER_SIZE_TWO; o++) {
        float sum = 0.0f;
        for(int i = 0; i < FC_LAYER_SIZE_ONE; i++) {
            sum += input[i] * fc1_weight[o * FC_LAYER_SIZE_ONE + i];
        }
        sum += fc1_bias[o];
        output[o] = sum;
    }
}

void dense_two(float * input, float * output) {
    for(int o = 0; o < FC_LAYER_SIZE_THREE; o++) {
        float sum = 0.0f;
        for(int i = 0; i < FC_LAYER_SIZE_TWO; i++) {
            sum += input[i] * fc2_weight[o * FC_LAYER_SIZE_TWO + i];
        }
        sum += fc2_bias[o];
        output[o] = sum;
    }
}

void dense_three(float * input, float * output) {
    for(int o = 0; o < FC_LAYER_SIZE_FOUR; o++) {
        float sum = 0.0f;
        for(int i = 0; i < FC_LAYER_SIZE_THREE; i++) {
            sum += input[i] * fc3_weight[o * FC_LAYER_SIZE_THREE + i];
        }
        sum += fc3_bias[o];
        output[o] = sum;
    }
}


