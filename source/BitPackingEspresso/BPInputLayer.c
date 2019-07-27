//
// Created by tianlun on 27/07/19.
//

#include "BitPackingEspresso/BPInputLayer.h"

void bp_input_layer_load(int D, int M, int N, int L,
        __uint8_t *in, BPInputLayer *il) {
//    if (il->in != NULL)
//        free(il->in);
    il->in = in;
    il->out = bp_tensor_init(D, M, N, L);
}

void bp_input_layer_free(BPInputLayer *il) {
    bp_tensor_free(&il->out);
}

void bp_input_layer_forward(BPInputLayer *il) {
    if (!il->out.data) {
        fprintf(stderr, "err: in null\n");
        exit(-1);
    }

    __uint8_t *input_data = il->in;
    const int len = bp_tensor_packed_len(&il->out);

    for (int i = 0; i < len; i++){
        __uint32_t tmp = 0;
        for (int j = 0; j < 32; j++){
            if (input_data[i * 32 + j] / 255 > 0.5)
                tmp = tmp | (1<<i);
        }
        il->out.data[i] = tmp;
    }
}


