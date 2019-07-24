#ifndef RANDOMINITIALISELAYER_H
#define RANDOMINITIALISELAYER_H

#include <stdlib.h>
#include "DenseLayer.h"
#include "BatchNormLayer.h"
#include "ConvolutionalLayer.h"

#define THRESHOLD 0.6


void dense_layer_rand_weight(DenseLayer *den_layer);

void bnorm_layer_rand_weight(bnormLayer *bnorm_layer);

void conv_layer_rand_weight(ConvLayer *conv_layer);

void random_init_arr(float *arr, size_t arr_length);

void random_init_tensor(int D, int M, int N, int L);

#endif //RANDOMINITIALISELAYER_H
