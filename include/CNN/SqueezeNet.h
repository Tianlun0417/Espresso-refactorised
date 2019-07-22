#ifndef SQUEEZENET_H
#define SQUEEZENET_H

#include "FloatTypeEspresso/FLOAT_ESP.h"
#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0

typedef FloatTensor Tensor;

typedef enum version{
    Version1_0,
    Version1_1
}SqueezeNetVersion;

typedef struct fire{
    int inplanes;
    ConvLayer * squeeze;
    ConvLayer * expand1x1;
    ConvLayer * expand3x3;
    Tensor    * output;
}Fire;

typedef struct featuresSequential{
    SqueezeNetVersion version;
    ConvLayer * conv;
    Fire      * fire_list[8];
    PoolLayer * maxpool_list[3];
    Tensor output;
}FeaturesSequential;

typedef struct classifierSequential{
    int num_classes;
    DropoutLayer * dropout;
    ConvLayer    * final_conv;
    PoolLayer    * avgpool;
    Tensor output;
}ClassifierSequential;

typedef struct squeezeNet{
    SqueezeNetVersion version;
    int num_classes;
    FeaturesSequential   * features;
    ClassifierSequential * classifier;
    Tensor output;
}SqueezeNet;

void fire_module_init(Fire *fire_ptr, int inplanes, int squeeze_planes,
                      int expand1x1_planes, int expand3x3_planes);
void features_sequential_init(FeaturesSequential *features_ptr, SqueezeNetVersion version);
void classifier_sequential_init(ClassifierSequential *classifier_ptr, int num_classes);
void SqueezeNet_init(SqueezeNet *squeeze_net_ptr, SqueezeNetVersion version, int num_classes);

void fire_forward(Tensor *input, Fire *fire);
void features_forward(Tensor * input, FeaturesSequential * features);
void classification_forward(Tensor * input, ClassifierSequential * classifier);
void squeezenet_forward(Tensor * input, SqueezeNet * squeeze_net);

void Squeezenet_free(SqueezeNet *squeeze_net);

#endif //ESPRESSO_REFACTORISED_SQUEEZENET_H
