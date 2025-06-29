#ifndef POOLING_H
#define POOLING_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "conv.h"

#define POOL_SIZE 2
#define POOL_STRIDE 2

#define POOL_INPUT_SIZE_ONE 24
#define POOL_INPUT_CHANNEL_ONE 6
#define POOL_OUTPUT_SIZE_ONE 12 //(24/2)
#define POOL_INPUT_SIZE_TWO 8
#define POOL_INPUT_CHANNEL_TWO 16
#define POOL_OUTPUT_SIZE_TWO 4 //(8/2)

void maxpool_one(float input[POOL_INPUT_CHANNEL_ONE][POOL_INPUT_SIZE_ONE][POOL_INPUT_SIZE_ONE], 
                 float output[POOL_INPUT_CHANNEL_ONE][POOL_OUTPUT_SIZE_ONE][POOL_OUTPUT_SIZE_ONE]);
                 
void maxpool_two(float input[POOL_INPUT_CHANNEL_TWO][POOL_INPUT_SIZE_TWO][POOL_INPUT_SIZE_TWO], 
                 float output[POOL_INPUT_CHANNEL_TWO][POOL_OUTPUT_SIZE_TWO][POOL_OUTPUT_SIZE_TWO]);

#endif