#ifndef DROPOUTLAYER_H
#define DROPOUTLAYER_H

#include "FloatTensor.h"
#include <stdlib.h>

typedef struct {
    float dropout_rate;
} DropoutLayer;

DropoutLayer * new_dropout_layer(float dropout_rate);

void dropout_layer_forward(FloatTensor *input_tensor, DropoutLayer *dropout_layer);

#endif //ESPRESSO_REFACTORISED_DROPOUTLAYER_H
