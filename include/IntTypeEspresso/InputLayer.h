#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "IntTensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    IntTensor out;
} InputLayer;

void inputLayer_load(IntTensor *in, InputLayer *il);

void inputLayer_free(InputLayer *il);

void inputLayer_forward(InputLayer *il);

void inputLayer_pad(InputLayer *il, int p);


#ifdef __cplusplus
}
#endif

#endif //ESPRESSO_REFACTORISED_INPUTLAYER_H
