#ifndef BPDROPOUTLAYER_H
#define BPDROPOUTLAYER_H

#include "BPTensor.h"
#include <stdlib.h>


typedef struct {
    float dropout_rate;
} BPDropoutLayer;

void bp_dropout_layer_init(BPDropoutLayer *dropout_layer, float dropout_rate);

void bp_dropout_layer_forward(BPTensor *input_tensor, BPDropoutLayer *dropout_layer);


#endif //ESPRESSO_REFACTORISED_BPDROPOUTLAYER_H
