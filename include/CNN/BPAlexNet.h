#ifndef BPALEXNET_H
#define BPALEXNET_H

#include "BitPackingEspresso/BP_ESP.h"
#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0


typedef BPTensor Tensor;

typedef struct features{
    BPConvLayer * conv1;
    BPConvLayer * conv2;
    BPConvLayer * conv3;
    BPConvLayer * conv4;
    BPConvLayer * conv5;
    BPPoolLayer * maxpool1;
    BPPoolLayer * maxpool2;
    BPPoolLayer * maxpool3;
    Tensor      output;
}Features;

typedef struct classifier{
    int num_classes;
    BPDropoutLayer * dropout;
    BPDenseLayer   * dense1;
    BPDenseLayer   * dense2;
    BPDenseOutputLayer   * dense3;
    float* output;
}Classifier;

typedef struct alexnet{
    int num_classes;
    Features   * features;
    Classifier * classifier;
    float*       output;
}BPAlexNet;

void BPAlexNet_init(BPAlexNet *alex_net, int num_classes);

void BPAlexNet_forward(Tensor *input, BPAlexNet *alex_net);

void BPAlexNet_free(BPAlexNet *alexnet);

#endif //ESPRESSO_REFACTORISED_BPALEXNET_H
