#ifndef INPUTLAYER_H
#define INPUTLAYER_H
#include "Tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    floatTensors out;
} inputLayer;

void inputLayer_load(floatTensors *in, inputLayer *il);
void inputLayer_free(inputLayer *il);
void inputLayer_forward(inputLayer *il);
void inputLayer_pad(inputLayer *il, int p);


#ifdef __cplusplus
}
#endif
#endif //INPUTLAYER_H
