#ifndef ALEXNET_H
#define ALEXNET_H

#include "FloatTypeEspresso/FLOAT_ESP.h"
#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0


typedef FloatTensor Tensor;

typedef struct features{
    convLayer * conv1;
    convLayer * conv2;
    convLayer * conv3;
    convLayer * conv4;
    convLayer * conv5;
    poolLayer * maxpool1;
    poolLayer * maxpool2;
    poolLayer * maxpool3;
    Tensor    * output;
}Features;

typedef struct classifier{
    int num_classes;
    dropoutLayer * dropout;
    denseLayer   * dense1;
    denseLayer   * dense2;
    denseLayer   * dense3;
    Tensor       * output;
}Classifier;

typedef struct alexnet{
    int num_classes;
    Features   * features;
    Classifier * classifier;
    Tensor     * output;
}AlexNet;

Features * new_features();
Classifier* new_classifier(int num_classes);
AlexNet * AlexNet_init(int num_classes);

void features_forward(Tensor *input, Features *features);
void classifier_forward(Tensor *input, Classifier *classifier);
void alexnet_forward(Tensor *input, AlexNet *alex_net);

#endif //ESPRESSO_REFACTORISED_ALEXNET_H
