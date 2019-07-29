#ifndef BPDENSENET_H
#define BPDENSENET_H

#include "BitPackingEspresso/BP_ESP.h"

#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0

typedef BPTensor Tensor;

typedef struct bpdensenetlayer{
    float drop_rate;
    BPBnormLayer   * bnorm1;
    BPBnormLayer   * bnorm2;
    BPConvLayer    * conv1;
    BPConvLayer    * conv2;
    BPDropoutLayer * dropout;
    Tensor output;
}BPDenseNetLayer;

typedef struct bptransition{
    BPBnormLayer * bnorm;
    BPConvLayer  * conv;
    BPPoolLayer  * pool;
    Tensor output;
}BPTransition;

typedef struct bpdenseblock{
    int num_layers;
    int num_input_features;
    int bn_size;
    int growth_rate;
    float drop_rate;
    BPDenseNetLayer ** densenet_layer_list;
    Tensor output;
}BPDenseBlock;

typedef struct bpdensenetfeatures{
    const int *block_config;
    int num_features;
    BPConvLayer  * conv0;
    BPBnormLayer * bnorm0;
    BPPoolLayer  * pool0;
    BPDenseBlock ** dense_block_list;
    BPTransition ** transition_list;
    BPBnormLayer * bnorm5;
    Tensor output;
}BPDenseNetFeatures;

typedef struct bpdensenet{
    BPDenseNetFeatures * features;
    BPDenseOutputLayer * classifier;
    BPPoolLayer  * avgpool;
    float* output;
}BPDenseNet;

//void bp_densenet_layer_init(BPDenseNetLayer *layer, int num_input_features, int bn_size,
//                         int growth_rate, float drop_rate);
//void bp_transition_init(BPTransition *transition, int num_input_faetures, int num_output_faetures);
//void bp_densenet_block_init(BPDenseBlock *block, int num_layers, int num_input_features,
//                         int bn_size, int growth_rate, float drop_rate);
//void bp_densenet_features_init(BPDenseNetFeatures *features, const int *block_config,
//                            int num_init_features, int growth_rate, int bn_size, float drop_rate);
void BPDenseNet_init(BPDenseNet *densenet, const int *block_config, int num_init_features,
                   int growth_rate, int bn_size, float drop_rate, int num_classes);

//void bp_densenet_layer_forward(Tensor *input, BPDenseNetLayer *layer);
//void bp_densenet_transition_forward(Tensor *input, BPTransition *transition);
//void bp_densenet_block_forward(Tensor *input, BPDenseBlock *block);
//void bp_densenet_features_forward(Tensor *input, BPDenseNetFeatures *features);
void BPDenseNet_forward(Tensor *input, BPDenseNet *densenet);

void BPDenseNet_free(BPDenseNet *densenet);



#endif //ESPRESSO_REFACTORISED_BPDENSENET_H
