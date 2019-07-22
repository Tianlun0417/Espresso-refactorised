#ifndef ALEXNET_H
#define ALEXNET_H

#include "FloatTypeEspresso/FLOAT_ESP.h"
#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0


typedef FloatTensor Tensor;

typedef struct features{
    ConvLayer * conv1;
    ConvLayer * conv2;
    ConvLayer * conv3;
    ConvLayer * conv4;
    ConvLayer * conv5;
    PoolLayer * maxpool1;
    PoolLayer * maxpool2;
    PoolLayer * maxpool3;
    Tensor    * output;
}Features;

typedef struct classifier{
    int num_classes;
    DropoutLayer * dropout;
    DenseLayer   * dense1;
    DenseLayer   * dense2;
    DenseLayer   * dense3;
    Tensor       * output;
}Classifier;

typedef struct alexnet{
    int num_classes;
    Features   * features;
    Classifier * classifier;
    Tensor     * output;
}AlexNet;

void new_features(Features *features);
void new_classifier(Classifier *classifier, int num_classes);
void AlexNet_init(AlexNet *alex_net, int num_classes);

void features_forward(Tensor *input, Features *features);
void classifier_forward(Tensor *input, Classifier *classifier);
void alexnet_forward(Tensor *input, AlexNet *alex_net);

#endif //ESPRESSO_REFACTORISED_ALEXNET_H
