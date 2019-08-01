#ifndef BPSQUEEZENET_H
#define BPSQUEEZENET_H

#include "BitPackingEspresso/BP_ESP.h"
#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0

typedef BPTensor Tensor;

typedef enum bp_version{
    Version1_0,
    Version1_1
}BPSqueezeNetVersion;

typedef struct bp_fire{
    int inplanes;
    BPConvLayer * squeeze;
    BPConvLayer * expand1x1;
    BPConvLayer * expand3x3;
    Tensor    * output;
}BPFire;

typedef struct bp_featuresSequential{
    BPSqueezeNetVersion version;
    BPConvLayer * conv;
    BPFire      * fire_list[8];
    BPPoolLayer * maxpool_list[3];
    Tensor output;
}BPFeaturesSequential;

typedef struct bp_classifierSequential{
    int num_classes;
    BPDropoutLayer * dropout;
    BPConvLayer    * final_conv;
    BPPoolLayer    * avgpool;
    Tensor output;
}BPClassifierSequential;

typedef struct bp_squeezeNet{
    BPSqueezeNetVersion version;
    int num_classes;
    BPFeaturesSequential   * features;
    BPClassifierSequential * classifier;
    Tensor output;
}BPSqueezeNet;


void bp_fire_module_init(BPFire *fire_ptr, int inplanes, int squeeze_planes,
                      int expand1x1_planes, int expand3x3_planes);
void bp_features_sequential_init(BPFeaturesSequential *features_ptr, BPSqueezeNetVersion version);
void bp_classifier_sequential_init(BPClassifierSequential *classifier_ptr, int num_classes);
void BPSqueezeNet_init(BPSqueezeNet *squeeze_net_ptr, BPSqueezeNetVersion version, int num_classes);

void bp_fire_forward(Tensor *input, BPFire *fire);
void bp_features_forward(Tensor * input, BPFeaturesSequential * features);
void bp_classification_forward(Tensor * input, BPClassifierSequential * classifier);
void BPSqueezeNet_forward(Tensor * input, BPSqueezeNet * squeeze_net);

void BPSqueezeNet_free(BPSqueezeNet *squeeze_net);


#endif //ESPRESSO_REFACTORISED_BPSQUEEZENET_H
