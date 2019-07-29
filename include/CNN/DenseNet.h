#ifndef DENSENET_H
#define DENSENET_H
#include "FloatTypeEspresso/FLOAT_ESP.h"

#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0

typedef FloatTensor Tensor;

typedef struct densenetlayer{
    float drop_rate;
    BnormLayer   * bnorm1;
    BnormLayer   * bnorm2;
    ConvLayer    * conv1;
    ConvLayer    * conv2;
    DropoutLayer * dropout;
    Tensor output;
}DenseNetLayer;

typedef struct transition{
    BnormLayer * bnorm;
    ConvLayer  * conv;
    PoolLayer  * pool;
    Tensor output;
}Transition;

typedef struct denseblock{
    int num_layers;
    int num_input_features;
    int bn_size;
    int growth_rate;
    float drop_rate;
    DenseNetLayer ** densenet_layer_list;
    Tensor output;
}DenseBlock;

typedef struct densenetfeatures{
    const int *block_config;
    int num_features;
    ConvLayer  * conv0;
    BnormLayer * bnorm0;
    PoolLayer  * pool0;
    DenseBlock ** dense_block_list;
    Transition ** transition_list;
    BnormLayer * bnorm5;
    Tensor output;
}DenseNetFeatures;

typedef struct densenet{
    DenseNetFeatures * features;
    DenseLayer * classifier;
    PoolLayer  * avgpool;
    Tensor output;
}DenseNet;

//void densenet_layer_init(DenseNetLayer *layer, int num_input_features, int bn_size,
//        int growth_rate, float drop_rate);
//void transition_init(Transition *transition, int num_input_faetures, int num_output_faetures);
//void densenet_block_init(DenseBlock *block, int num_layers, int num_input_features,
//                         int bn_size, int growth_rate, float drop_rate);
//void densenet_features_init(DenseNetFeatures *features, const int *block_config,
//                            int num_init_features, int growth_rate, int bn_size, float drop_rate);
void DenseNet_init(DenseNet *densenet, const int *block_config, int num_init_features,
        int growth_rate, int bn_size, float drop_rate, int num_classes);

//void densenet_layer_forward(Tensor *input, DenseNetLayer *layer);
//void densenet_transition_forward(Tensor *input, Transition *transition);
//void densenet_block_forward(Tensor *input, DenseBlock *block);
//void densenet_features_forward(Tensor *input, DenseNetFeatures *features);
void DenseNet_forward(Tensor *input, DenseNet *densenet);

void DenseNet_free(DenseNet *densenet);


#endif //ESPRESSO_REFACTORISED_DENSENET_H
