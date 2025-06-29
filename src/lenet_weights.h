#ifndef LENET_WEIGHTS_H
#define LENET_WEIGHTS_H

// Use 'extern' to declare the variables without defining them.
// This tells the compiler that these variables exist but are defined in another source file.

// Convolutional Layer 1
extern const float conv1_weight[150];
extern const float conv1_bias[6];

// Convolutional Layer 2
extern const float conv2_weight[2400]; // 16 * 6 * 5 * 5

// Fully Connected Layer 1
extern const float conv2_bias[16];
extern const float fc1_weight[30720]; // 120 * 256
extern const float fc1_bias[120];

// Fully Connected Layer 2
extern const float fc2_weight[10080]; // 84 * 120
extern const float fc2_bias[84];

// Fully Connected Layer 3 (Output)
extern const float fc3_weight[840]; // 10 * 84
extern const float fc3_bias[10];

#endif // LENET_WEIGHTS_H
