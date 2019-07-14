#include "DropoutLayer.h"


dropoutLayer dropoutLayer_init(float dropout_rate){
    dropoutLayer dropout_layer;
    dropout_layer.dropout_rate = dropout_rate;
    return dropout_layer;
}

void dropoutLayer_forward(FloatTensor * input_tensor, dropoutLayer * dropout_layer){
    for(int i=0; i<input_tensor->MNL; i++){
        if((float) rand()/(float) (RAND_MAX)>dropout_layer->dropout_rate)
            input_tensor->data[i] = 0;
    }
}