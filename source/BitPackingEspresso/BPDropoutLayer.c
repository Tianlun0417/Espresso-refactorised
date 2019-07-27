#include "BitPackingEspresso/BPDropoutLayer.h"

void bp_dropout_layer_init(BPDropoutLayer *dropout_layer, float dropout_rate) {
    dropout_layer->dropout_rate = dropout_rate;
}

void bp_dropout_layer_forward(BPTensor *input_tensor, BPDropoutLayer *dropout_layer) {
    __uint32_t dropout_mask = 0;
    size_t packed_size = (input_tensor->D * input_tensor->MNL) / 32;

    for (int i = 0; i < 32; i++) {
        if ((float) rand() / (float) (RAND_MAX) > dropout_layer->dropout_rate)
            dropout_mask = dropout_mask | (1<<i);
    }

    for (int i = 0; i < packed_size; i++)
        input_tensor->data[i] = input_tensor->data[i] & dropout_mask;
}
