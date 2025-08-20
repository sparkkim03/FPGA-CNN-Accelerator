#ifndef DENSE_H
#define DENSE_H

// We will start working with flattened arrays
#define FC_LAYER_SIZE_ONE 256
#define FC_LAYER_SIZE_TWO 120
#define FC_LAYER_SIZE_THREE 84
#define FC_LAYER_SIZE_FOUR 10

void dense(float *input, int input_size, 
           float *output, int output_size, 
           const float *weights, const float *biases);

#endif