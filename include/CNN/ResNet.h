#ifndef RESNET_H
#define RESNET_H

#include "FloatTypeEspresso/FLOAT_ESP.h"
#include "DataLoader/Cifar10Loader.h"
#include <stdbool.h>

#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0

typedef FloatTensor Tensor;

typedef enum BlockType {
    UseBasicBlock,
    UseBottleneck
} BlockType;

typedef struct down_sample {
    ConvLayer *conv;
    bnormLayer *bn;
    Tensor output;
} Downsample;

typedef struct basic {
    int expansion;
    int stride;
    ConvLayer *conv1;
    bnormLayer *bn1;
    ConvLayer *conv2;
    bnormLayer *bn2;
    Downsample *downsample;
    Tensor residual;
    Tensor output;
} BasicBlock;

typedef struct bottleneck {
    int expansion;
    int stride;
    ConvLayer *conv1;
    bnormLayer *bn1;
    ConvLayer *conv2;
    bnormLayer *bn2;
    ConvLayer *conv3;
    bnormLayer *bn3;
    Downsample *downsample;
    Tensor residual;
    Tensor output;
} Bottleneck;

typedef struct resnet_block {
    int num_blocks;
    BlockType block_type;
    BasicBlock **basicblocks;
    Bottleneck **bottlenecks;
    Tensor output;
} ResNetBlock;

typedef struct resnet {
    int inplanes;
    BlockType block_type;
    ConvLayer *conv1;
    bnormLayer *bn1;
    PoolLayer *pool1;
    ResNetBlock *block1;
    ResNetBlock *block2;
    ResNetBlock *block3;
    ResNetBlock *block4;
    PoolLayer *pool2;
    DenseLayer *fc;
    Tensor output;
} ResNet;

void ResNet_init(ResNet *ResNetInstance, BlockType block_type, int num_layers[4],
                 int num_classes);
void ResNet_block_init(ResNetBlock *block_ptr, ResNet *resnet_ptr, int planes,
                       int num_blocks, int stride);
void basicblock_init(BasicBlock *basicblock, int inplanes, int planes,
                     int stride, Downsample *downsample);
void bottleneck_init(Bottleneck *bottleneck, int inplanes, int planes,
                     int stride, Downsample *downsample);

void downsample_forward(Tensor * input, Downsample * downsample);
void basicblock_forward(Tensor * input, BasicBlock * basicblock);
void bottleneck_forward(Tensor * input, Bottleneck * bottleneck);
void resnet_block_forward(Tensor * input, ResNetBlock * block);
void resnet_forward(Tensor *image_tensor, ResNet *resnet);

void ResNet_free(ResNet * resnet);
#endif //ESPRESSO_REFACTORISED_RESNET_H
