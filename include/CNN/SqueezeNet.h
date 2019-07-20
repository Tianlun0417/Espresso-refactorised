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
    convLayer * squeeze;
    convLayer * expand1x1;
    convLayer * expand3x3;
    Tensor    * output;
}Fire;

typedef struct featuresSequential{
    SqueezeNetVersion version;
    convLayer * conv;
    Fire      * fire_list[8];
    poolLayer * maxpool_list[3];
    Tensor * output;
}FeaturesSequential;

typedef struct classifierSequential{
    int num_classes;
    dropoutLayer * dropout;
    convLayer    * final_conv;
    poolLayer    * avgpool;
    Tensor * output;
}ClassifierSequential;

typedef struct squeezeNet{
    SqueezeNetVersion version;
    int num_classes;
    FeaturesSequential   * features;
    ClassifierSequential * classifier;
    Tensor * output;
}SqueezeNet;

Fire * new_fire_module(int inplanes, int squeeze_planes,
        int expand1x1_planes, int expand3x3_planes);
FeaturesSequential * new_features_sequential(SqueezeNetVersion version);
ClassifierSequential * new_classifier_sequential(int num_classes);
SqueezeNet * SqueezeNet_init(SqueezeNetVersion version, int num_classes);

void fire_forward(Tensor * input, Fire * fire);
void features_forward(Tensor * input, FeaturesSequential * features);
void classification_forward(Tensor * input, ClassifierSequential * classifier);
void squeezenet_forward(Tensor * input, SqueezeNet * squeeze_net);

#endif //ESPRESSO_REFACTORISED_SQUEEZENET_H
