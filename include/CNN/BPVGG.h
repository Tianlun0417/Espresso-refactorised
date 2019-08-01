//
// Created by tianlun on 01/08/19.
//

#ifndef BPVGG_H
#define BPVGG_H

#include "BitPackingEspresso/BP_ESP.h"
#include <stdbool.h>

#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0
#define M -1

typedef BPTensor Tensor;

typedef enum bp_vggconfig{
    ConfigA,
    ConfigB,
    ConfigD,
    ConfigE
}BPVGGConfig;

typedef struct bp_vgg_features{
    int conv_count;
    bool batch_norm;
    BPVGGConfig  config;
    BPConvLayer  ** conv_list;
    BPPoolLayer  *  maxpool_list[5];
    BPBnormLayer ** bnorm_list;
    Tensor       output;
}BPVGGFeatures;

typedef struct bp_vgg_classifier{
    BPDenseLayer       * dense1;
    BPDropoutLayer     * dropout1;
    BPDenseLayer       * dense2;
    BPDropoutLayer     * dropout2;
    BPDenseOutputLayer * dense3;
    float *output;
}BPVGGClassifier;

typedef struct bp_vgg{
    bool batch_norm;
    BPVGGConfig       config;
    BPVGGFeatures   * features;
    BPVGGClassifier * classifier;
    float *output;
}BPVGG;

void bp_features_init(BPVGGFeatures *features, bool batch_norm, BPVGGConfig config);
void bp_classifier_init(BPVGGClassifier *classifier, int num_classes);
void BPVGG_init(BPVGG *vgg, BPVGGConfig config, int num_classes, bool batch_norm);

void bp_features_forward(Tensor *input, BPVGGFeatures *features);
void bp_classifier_forward(Tensor *input, BPVGGClassifier *classifier);
void BPVGG_forward(Tensor *input, BPVGG *vgg);

void BPVGG_free(BPVGG *vgg);

#endif //ESPRESSO_REFACTORISED_BPVGG_H
