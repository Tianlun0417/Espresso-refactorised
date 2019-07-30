#ifndef BPRESNET_H
#define BPRESNET_H

#include "BitPackingEspresso/BP_ESP.h"
#include <stdbool.h>

#define SAVE 1
#define LOAD_PRETRAINED_WEIGHT 0

typedef BPTensor Tensor;

typedef enum BP_BlockType {
    UseBasicBlock,
    UseBottleneck
} BPBlockType;

typedef struct bp_down_sample {
    BPConvLayer *conv;
    BPBnormLayer *bn;
    Tensor output;
} BPDownsample;

typedef struct bp_basic {
    int expansion;
    int stride;
    BPConvLayer *conv1;
    BPBnormLayer *bn1;
    BPConvLayer *conv2;
    BPBnormLayer *bn2;
    BPDownsample *downsample;
    Tensor residual;
    Tensor output;
} BPBasicBlock;

typedef struct bp_bottleneck {
    int expansion;
    int stride;
    BPConvLayer *conv1;
    BPBnormLayer *bn1;
    BPConvLayer *conv2;
    BPBnormLayer *bn2;
    BPConvLayer *conv3;
    BPBnormLayer *bn3;
    BPDownsample *downsample;
    Tensor residual;
    Tensor output;
} BPBottleneck;

typedef struct bp_resnet_block {
    int num_blocks;
    BPBlockType block_type;
    BPBasicBlock **basicblocks;
    BPBottleneck **bottlenecks;
    Tensor output;
} BPResNetBlock;

typedef struct bp_resnet {
    int inplanes;
    BPBlockType block_type;
    BPConvLayer *conv1;
    BPBnormLayer *bn1;
    BPPoolLayer *pool1;
    BPResNetBlock *block1;
    BPResNetBlock *block2;
    BPResNetBlock *block3;
    BPResNetBlock *block4;
    BPPoolLayer *pool2;
    BPDenseOutputLayer *fc;
    float *output;
} BPResNet;

void BPResNet_init(BPResNet *ResNetInstance, BPBlockType block_type, int num_layers[4],
                 int num_classes);
//void bp_basicblock_init(BPBasicBlock *basicblock, int inplanes, int planes,
//                     int stride, BPDownsample *downsample);
//void bp_bottleneck_init(BPBottleneck *bottleneck, int inplanes, int planes,
//                     int stride, BPDownsample *downsample);
//void bp_resnet_block_init(BPResNetBlock *block_ptr, BPResNet *resnet_ptr, int planes,
//                          int num_blocks, int stride);

//void bp_downsample_forward(Tensor * input, BPDownsample * downsample);
//void bp_basicblock_forward(Tensor * input, BPBasicBlock * basicblock);
//void bp_bottleneck_forward(Tensor * input, BPBottleneck * bottleneck);
//void bp_resnet_block_forward(Tensor * input, BPResNetBlock * block);
void BPResNet_forward(Tensor *image_tensor, BPResNet *resnet);

void BPResNet_free(BPResNet * resnet);

#endif //ESPRESSO_REFACTORISED_BPRESNET_H
