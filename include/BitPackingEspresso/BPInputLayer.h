#ifndef BPINPUTLAYER_H
#define BPINPUTLAYER_H

#include "BPTensor.h"
#include <stdlib.h>


typedef struct {
    __uint8_t *in;
    BPTensor out;
} BPInputLayer;
//
//void bp_input_layer_load(int D, int M, int N, int L, __uint8_t *in, BPInputLayer *il);
//
//void bp_input_layer_free(BPInputLayer *il);

void bp_input_layer_forward(const __uint8_t *in, BPTensor *out);


#endif //ESPRESSO_REFACTORISED_BPINPUTLAYER_H
