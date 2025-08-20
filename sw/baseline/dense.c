#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "dense.h"
#include "lenet_weights.h"

void dense(float *input, int input_size, 
           float *output, int output_size, 
           const float *weights, const float *biases) {

    for (int o = 0; o < output_size; o++) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            // Access the weight from the 1D array using the output and input indices
            sum += input[i] * weights[o * input_size + i];
        }
        sum += biases[o];
        output[o] = sum;
    }
}

