#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "FloatTensor.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    FloatTensor out;
} InputLayer;

void input_layer_load(FloatTensor *in, InputLayer *il);

void input_layer_free(InputLayer *il);

void input_layer_forward(InputLayer *il);

void inputLayer_pad(InputLayer *il, int p);


#ifdef __cplusplus
}
#endif
#endif //INPUTLAYER_H
