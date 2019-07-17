#ifndef RESNET_H
#define RESNET_H

#include "FloatTypeEspresso/FLOAT_ESP.h"
#include "FloatTypeEspresso/Cifar10Loader.h"
#include <stdbool.h>

#define SAVE 1

typedef FloatTensor Tensor;

typedef enum BlockType {
    UseBasicBlock,
    UseBottleneck
} BlockType;

typedef struct down_sample {
    convLayer *conv;
    bnormLayer *bn;
    Tensor *output;
} Downsample;

typedef struct basic {
    int expansion;
    int stride;
    convLayer *conv1;
    bnormLayer *bn1;
    convLayer *conv2;
    bnormLayer *bn2;
    Downsample *downsample;
    Tensor *output;
} BasicBlock;

typedef struct bottleneck {
    int expansion;
    int stride;
    convLayer *conv1;
    bnormLayer *bn1;
    convLayer *conv2;
    bnormLayer *bn2;
    convLayer *conv3;
    bnormLayer *bn3;
    Downsample *downsample;
    Tensor *output;
} Bottleneck;

typedef struct resnet_block {
    int num_blocks;
    BlockType block_type;
    BasicBlock **basicblocks;
    Bottleneck **bottlenecks;
    Tensor *output;
} ResNetBlock;

typedef struct resnet {
    int inplanes;
    BlockType block_type;
    convLayer *conv1;
    bnormLayer *bn1;
    poolLayer *pool1;
    ResNetBlock *block1;
    ResNetBlock *block2;
    ResNetBlock *block3;
    ResNetBlock *block4;
    poolLayer *pool2;
    denseLayer *fc;
    Tensor *output;
} ResNet;

ResNet *ResNet_init(BlockType block_type, int num_layers[4], int num_classes);

//ResNetBlock * build_ResNet_block(ResNet * resnet, int planes, int num_blocks, int stride);
//BasicBlock * new_basicblock(int inplanes, int planes, int stride, Downsample * downsample);
//Bottleneck * new_bottleneck(int inplanes, int planes, int stride, Downsample * downsample);
//convLayer * new_conv_layer(int D, int M, int N, int L, int Stride_m, int Stride_n, int padding);
//bnormLayer * new_bn_layer(int size);
//poolLayer * new_pool_layer(int M, int N, int Stride_m, int Stride_n, poolingStrategy strategy);
//denseLayer * new_dense_layer(int M, int N);
//void downsample_forward(Tensor * input, Downsample * downsample);
//void basicblock_forward(Tensor * input, BasicBlock * basicblock);
//void bottleneck_forward(Tensor * input, Bottleneck * bottleneck);
//void resnet_block_forward(Tensor * input, ResNetBlock * block);
void resnet_forward(Tensor *image_tensor, ResNet *resnet);

#endif //ESPRESSO_REFACTORISED_RESNET_H
