#ifndef DENSE_H
#define DENSE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// We will start working with flattened arrays
#define FC_LAYER_SIZE_ONE 256
#define FC_LAYER_SIZE_TWO 120
#define FC_LAYER_SIZE_THREE 84
#define FC_LAYER_SIZE_FOUR 10

void dense_one(float * input, float * output);
void dense_two(float * input, float * output);
void dense_three(float * input, float * output);
void dense_four(float * input, float * output);

#endif