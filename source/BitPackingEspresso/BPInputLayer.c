//
// Created by tianlun on 27/07/19.
//

#include "BitPackingEspresso/BPInputLayer.h"


void bp_input_layer_forward(const __uint8_t *in, BPTensor *out) {
    for (int i = 0; i < out->packed_len; i++){
        __uint32_t tmp = 0;
        for (int j = 0; j < 32; j++){
            if (in[i * 32 + j] / 255 > 0.5)
                tmp = tmp | (1<<i);
        }
        out->data[i] = tmp;
    }
}


