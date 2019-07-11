#ifndef RANDOMINITIALISELAYER_H
#define RANDOMINITIALISELAYER_H

#include <stdlib.h>
#include "DenseLayer.h"
#include "BatchNormLayer.h"
#include "ConvolutionalLayer.h"
#define THRESHOLD 0.2


void init_dense_layer(denseLayer *den_layer, int M, int N);
void init_batchnorm_layer(bnormLayer *bnorm_layer, size_t layer_size);
void init_conv_layer(convLayer *conv_layer, int D, int M, int N, int L);
void random_init_arr(float* arr, size_t arr_length);
void random_init_tensor(int D, int M, int N, int L);

#endif //RANDOMINITIALISELAYER_H
