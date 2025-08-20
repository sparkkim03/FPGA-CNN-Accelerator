#ifndef POOLING_H
#define POOLING_H

#define POOL_SIZE 2
#define POOL_STRIDE 2

#define POOL_INPUT_SIZE_ONE 24
#define POOL_INPUT_CHANNEL_ONE 6
#define POOL_OUTPUT_SIZE_ONE 12 //(24/2)
#define POOL_INPUT_SIZE_TWO 8
#define POOL_INPUT_CHANNEL_TWO 16
#define POOL_OUTPUT_SIZE_TWO 4 //(8/2)

void maxpool(const float* input, int input_channels, int input_size,
             float* output, int output_size);

#endif