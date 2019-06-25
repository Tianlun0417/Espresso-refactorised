#ifndef RANDOMINITIALISELAYER_H
#define RANDOMINITIALISELAYER_H

#include <stdlib.h>
#include "DenseLayer.h"
#include "BatchNormLayer.h"
#include "ConvolutionalLayer.h"

void init_float_tensor(int D, int M, int N, int L);
void init_dense_layer(denseLayer *den_layer);
void init_batchnorm_layer(bnormLayer *batchnorm_layer);
void init_conv_layer(convLayer *conv_layer);

#endif //RANDOMINITIALISELAYER_H
