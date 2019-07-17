#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "IntTensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    IntTensor out;
} inputLayer;

void inputLayer_load(IntTensor *in, inputLayer *il);

void inputLayer_free(inputLayer *il);

void inputLayer_forward(inputLayer *il);

void inputLayer_pad(inputLayer *il, int p);


#ifdef __cplusplus
}
#endif

#endif //ESPRESSO_REFACTORISED_INPUTLAYER_H
