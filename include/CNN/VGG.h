#ifndef VGG_H
#define VGG_H

#include "FloatTypeEspresso/FLOAT_ESP.h"
#include <stdbool.h>

#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0
#define M -1

typedef FloatTensor Tensor;

typedef enum vggconfig{
    ConfigA,
    ConfigB,
    ConfigD,
    ConfigE
}VGGConfig;

typedef struct vgg_features{
    int conv_count;
    bool batch_norm;
    VGGConfig  config;
    ConvLayer  ** conv_list;
    PoolLayer  *  maxpool_list[5];
    bnormLayer ** bnorm_list;
    Tensor       output;
}VGGFeatures;

typedef struct vgg_classifier{
    DenseLayer   * dense1;
    DropoutLayer * dropout1;
    DenseLayer   * dense2;
    DropoutLayer * dropout2;
    DenseLayer   * dense3;
    Tensor output;
}VGGClassifier;

typedef struct vgg{
    bool batch_norm;
    VGGConfig       config;
    VGGFeatures   * features;
    VGGClassifier * classifier;
    Tensor output;
}VGG;

void features_init(VGGFeatures *features, bool batch_norm, VGGConfig config);
void classifier_init(VGGClassifier *classifier, int num_classes);
void VGG_init(VGG *vgg, VGGConfig config, int num_classes, bool batch_norm);

void features_forward(Tensor *input, VGGFeatures *features);
void classifier_forward(Tensor *input, VGGClassifier *classifier);
void VGG_forward(Tensor *input, VGG *vgg);

void VGG_free(VGG *vgg);

#endif //ESPRESSO_REFACTORISED_VGG_H
