#ifndef INPUTLAYER_H
#define INPUTLAYER_H
#include "FloatTensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    FloatTensor out;
} inputLayer;

void inputLayer_load(FloatTensor *in, inputLayer *il);
void inputLayer_free(inputLayer *il);
void inputLayer_forward(inputLayer *il);
void inputLayer_pad(inputLayer *il, int p);


#ifdef __cplusplus
}
#endif
#endif //INPUTLAYER_H
