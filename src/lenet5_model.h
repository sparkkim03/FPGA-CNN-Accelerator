#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#ifndef LENET5_MODEL_H
#define LENET5_MODEL_H
// ________________________________________________________________________
// HYPERPARAMETERS & MODEL CONFIGURATION
// ________________________________________________________________________
#define MNIST_WIDTH 28
#define MNIST_HEIGHT 28
#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define INPUT_CHANNELS 1

#define NORM_MEAN 0.1307f
#define NORM_STD_DEV 0.3081f

// Layer 1
#define C1_FILTERS 6
#define C1_KERNEL_SIZE 5
#define C1_OUTPUT_WIDTH 28 
#define C1_OUTPUT_HEIGHT 28

// Layer 2
#define S2_POOL_SIZE 2
#define S2_STRIDE 2
#define S2_OUTPUT_WIDTH 14 
#define S2_OUTPUT_HEIGHT 14

// Layer 3
#define C3_FILTERS 16
#define C3_KERNEL_SIZE 5
#define C3_OUTPUT_WIDTH 10 
#define C3_OUTPUT_HEIGHT 10

// Layer 4
#define S4_POOL_SIZE 2
#define S4_STRIDE 2
#define S4_OUTPUT_WIDTH 5 
#define S4_OUTPUT_HEIGHT 5

// Flattening
#define FLATTEN_SIZE (C3_FILTERS * S4_OUTPUT_WIDTH * S4_OUTPUT_HEIGHT) 

// Layer 5
#define F5_OUTPUTS 120

// Layer 6
#define F6_OUTPUTS 84

// Layer 7
#define OUTPUT_SIZE 10

// ________________________________________________________________________
// MODEL WEIGHTS
// ________________________________________________________________________

extern const float c1_weights[C1_FILTERS][INPUT_CHANNELS][C1_KERNEL_SIZE][C1_KERNEL_SIZE];
extern const float c1_biases[C1_FILTERS];
extern const float c3_weights[C3_FILTERS][C1_FILTERS][C3_KERNEL_SIZE][C3_KERNEL_SIZE];
extern const float c3_biases[C3_FILTERS];
extern const float f5_weights[F5_OUTPUTS][FLATTEN_SIZE];
extern const float f5_biases[F5_OUTPUTS];
extern const float f6_weights[F6_OUTPUTS][F5_OUTPUTS];
extern const float f6_biases[F6_OUTPUTS];
extern const float output_weights[OUTPUT_SIZE][F6_OUTPUTS];
extern const float output_biases[OUTPUT_SIZE];

// ________________________________________________________________________
// Activation Buffers
// ________________________________________________________________________

extern float input_padded[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];
extern float c1_output[C1_FILTERS][C1_OUTPUT_HEIGHT][C1_OUTPUT_WIDTH];
extern float s2_output[C1_FILTERS][S2_OUTPUT_HEIGHT][S2_OUTPUT_WIDTH];
extern float c3_output[C3_FILTERS][C3_OUTPUT_HEIGHT][C3_OUTPUT_WIDTH];
extern float s4_output[C3_FILTERS][S4_OUTPUT_HEIGHT][S4_OUTPUT_WIDTH];
extern float flattened_output[FLATTEN_SIZE];
extern float f5_output[F5_OUTPUTS];
extern float f6_output[F6_OUTPUTS];
extern float final_output[OUTPUT_SIZE];

#endif