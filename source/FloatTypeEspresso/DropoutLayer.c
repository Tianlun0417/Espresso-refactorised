#include "FloatTypeEspresso/DropoutLayer.h"


DropoutLayer * new_dropout_layer(float dropout_rate) {
    DropoutLayer * dropout_layer = (DropoutLayer*) malloc(sizeof(DropoutLayer));
    dropout_layer->dropout_rate = dropout_rate;
    return dropout_layer;
}

void dropout_layer_forward(FloatTensor *input_tensor, DropoutLayer *dropout_layer) {
    for (int i = 0; i < tensor_len(input_tensor); i++) {
        if ((float) rand() / (float) (RAND_MAX) > dropout_layer->dropout_rate)
            input_tensor->data[i] = 0;
    }
}