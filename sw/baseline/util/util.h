#ifndef UTIL_H
#define UTIL_H

#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define BMP_HEADER_SIZE 54

void load_image(const char * filename, float img_array[IMG_HEIGHT][IMG_WIDTH]);

void flatten(float input[16][4][4], float * output); 

int argmax(float * input, int size);

#endif 