#ifndef RESNET_H
#define RESNET_H
#include "FloatTypeEspresso/ESP_RE.h"
#include "FloatTypeEspresso/Cifar10Loader.h"
#include <stdbool.h>


typedef enum BlockType{
    BasicBlock,
    Bottleneck
}BlockType;

typedef struct down_sample{
    convLayer  * conv;
    bnormLayer * bn;
}Downsample;

typedef struct resnet_block{
    int expansion;
    enum BlockType block_type;
    convLayer  * conv_layer_ptr;
    bnormLayer * bn_layer_ptr;
    Downsample * downsample_ptr;
}ResNetBlock;

typedef struct resnet{
    int inplanes;
    convLayer   * conv1;
    bnormLayer  * bn1;
    poolLayer   * maxpool;
    ResNetBlock * block1;
    ResNetBlock * block2;
    ResNetBlock * block3;
    ResNetBlock * block4;
    poolLayer   * avgpool;
    denseLayer  * fc;
}ResNet;

ResNet ResNet_init(BlockType block_type, int num_layers[4], int num_classes);
ResNetBlock* build_ResNet_block(BlockType block_type, int planes, int blocks, int stride);
convLayer* new_conv_layer(int D, int M, int N, int L,
                          int Stride_m, int Stride_n,
                          int padding);
bnormLayer* new_bn_layer();
poolLayer* new_pool_layer(int M, int N, int Stride_m,
        int Stride_n, poolingStrategy strategy);

#endif //ESPRESSO_REFACTORISED_RESNET_H
