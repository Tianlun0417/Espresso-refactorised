#ifndef DROPOUTLAYER_H
#define DROPOUTLAYER_H

#include "FloatTensor.h"
#include <stdlib.h>

typedef struct {
    float dropout_rate;
} dropoutLayer;

dropoutLayer * new_dropout_layer(float dropout_rate);

void dropoutLayer_forward(FloatTensor *input_tensor, dropoutLayer *dropout_layer);

#endif //ESPRESSO_REFACTORISED_DROPOUTLAYER_H
