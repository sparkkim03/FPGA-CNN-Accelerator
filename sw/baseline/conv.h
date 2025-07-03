#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#define FILTER_SIZE 5 // kernel is 5x5
#define NUM_FILTER_ONE 6 // first CONV layer has 6 filters
#define NUM_FILTER_TWO 16 // second CONV layer has 16 filters
#define INPUT_SIZE_ONE 28
#define INPUT_SIZE_TWO 12
#define OUTPUT_SIZE_ONE 24
#define OUTPUT_SIZE_TWO 8
#define CHANNEL_ONE 1 // number of channel in the first conv
#define CHANNEL_TWO 6 // number of channels in the second conv

typedef struct {
    float weights[FILTER_SIZE][FILTER_SIZE];
    float bias;
} Filter;

void initFilter_One(Filter * filter, int filterNth);
void initFilter_Two(Filter * filter, int filtherNth);

void convolve_one(float input[INPUT_SIZE_ONE][INPUT_SIZE_ONE], 
                    float output[NUM_FILTER_ONE][OUTPUT_SIZE_ONE][OUTPUT_SIZE_ONE],
                    Filter filters[NUM_FILTER_ONE]
                );
void convolve_two(float input[CHANNEL_TWO][INPUT_SIZE_TWO][INPUT_SIZE_TWO], 
                    float output[NUM_FILTER_TWO][OUTPUT_SIZE_TWO][OUTPUT_SIZE_TWO],
                    Filter filters[NUM_FILTER_TWO * CHANNEL_TWO]
                );

void reLU(float * input, int inputSize);

#endif