#include "CNN/ResNet.h"


void basicblock_init(BasicBlock *basicblock, int inplanes, int planes,
                     int stride, Downsample *downsample) {
    basicblock->expansion = 1;
    basicblock->stride = stride;
    basicblock->conv1 = malloc(sizeof(ConvLayer));
    basicblock->conv2 = malloc(sizeof(ConvLayer));
    basicblock->bn1 = new_bn_layer(planes);
    basicblock->bn2 = new_bn_layer(planes);
    basicblock->downsample = downsample;

    conv_layer_init(basicblock->conv1, inplanes, planes, 3, 3, stride, stride, 1);
    conv_layer_init(basicblock->conv2, planes, planes, 3, 3, stride, stride, 1);

    if(!LOAD_PRETRAINED_WEIGHT){
        conv_layer_rand_weight(basicblock->conv1);
        batchnorm_layer_rand_weight(basicblock->bn1, planes);
        conv_layer_rand_weight(basicblock->conv2);
        batchnorm_layer_rand_weight(basicblock->bn2, planes);
    }
}

void bottleneck_init(Bottleneck *bottleneck, int inplanes, int planes,
                     int stride, Downsample *downsample) {
    bottleneck->expansion = 4;
    bottleneck->stride = stride;
    bottleneck->conv1 = malloc(sizeof(ConvLayer));
    bottleneck->conv2 = malloc(sizeof(ConvLayer));
    bottleneck->conv3 = malloc(sizeof(ConvLayer));
    bottleneck->bn1 = new_bn_layer(planes);
    bottleneck->bn2 = new_bn_layer(planes);
    bottleneck->bn3 = new_bn_layer(planes * 4);
    bottleneck->downsample = downsample;

    conv_layer_init(bottleneck->conv1, inplanes, planes, 1, 1, 1, 1, 0);
    conv_layer_init(bottleneck->conv2, planes, planes, 3, 3, stride, stride, 1);
    conv_layer_init(bottleneck->conv3, planes, planes * 4, 1, 1, 1, 1, 0);

    if(!LOAD_PRETRAINED_WEIGHT){
        conv_layer_rand_weight(bottleneck->conv1);
        batchnorm_layer_rand_weight(bottleneck->bn1, planes);
        conv_layer_rand_weight(bottleneck->conv2);
        batchnorm_layer_rand_weight(bottleneck->bn2, planes);
        conv_layer_rand_weight(bottleneck->conv3);
        batchnorm_layer_rand_weight(bottleneck->bn3, planes * 4);
    }
}

void ResNet_block_init(ResNetBlock *block_ptr, ResNet *resnet_ptr, int planes,
                       int num_blocks, int stride) {
    block_ptr->num_blocks = num_blocks;
    block_ptr->block_type = resnet_ptr->block_type;

    Downsample *downsample_ptr = NULL;
    int expansion = resnet_ptr->block_type == UseBasicBlock ? 1 : 4;

    if (stride != 1 || resnet_ptr->inplanes != planes * expansion) {
        downsample_ptr = malloc(sizeof(Downsample));
        downsample_ptr->conv = malloc(sizeof(ConvLayer));
        conv_layer_init(downsample_ptr->conv, resnet_ptr->inplanes, planes, 1, 1, stride, stride, 0);
        downsample_ptr->bn = new_bn_layer(planes * expansion);

        if(!LOAD_PRETRAINED_WEIGHT){
            conv_layer_rand_weight(downsample_ptr->conv);
            batchnorm_layer_rand_weight(downsample_ptr->bn, planes * expansion);
        }
    }

    if (resnet_ptr->block_type == UseBasicBlock) {
        block_ptr->bottlenecks = NULL;
        // Init the memory for basic blocks
        block_ptr->basicblocks =
                (BasicBlock**) malloc((num_blocks + 1) * sizeof(BasicBlock*));
        // Init each basic block
        block_ptr->basicblocks[0] = malloc(sizeof(BasicBlock));
        basicblock_init(block_ptr->basicblocks[0], resnet_ptr->inplanes, planes, stride, downsample_ptr);
        resnet_ptr->inplanes = planes * 1;
        for (int block_idx = 1; block_idx < num_blocks + 1; block_idx++) {
            block_ptr->basicblocks[block_idx] = malloc(sizeof(BasicBlock));
            basicblock_init(block_ptr->basicblocks[block_idx], resnet_ptr->inplanes, planes, 1, NULL);
        }
    } else if (resnet_ptr->block_type == UseBottleneck) {
        block_ptr->basicblocks = NULL;
        block_ptr->bottlenecks =
                (Bottleneck**) malloc((num_blocks + 1) * sizeof(BasicBlock*));
        block_ptr->bottlenecks[0] = malloc(sizeof(BasicBlock));
        bottleneck_init(block_ptr->bottlenecks[0], resnet_ptr->inplanes, planes, stride, downsample_ptr);
        resnet_ptr->inplanes = planes * 4;
        for (int block_idx = 1; block_idx < num_blocks + 1; block_idx++) {
            block_ptr->bottlenecks[block_idx] = malloc(sizeof(BasicBlock));
            bottleneck_init(block_ptr->bottlenecks[block_idx], resnet_ptr->inplanes, planes, 1, NULL);
        }
    } else {
        puts("Non-supported block type! Use BasicBlock or Bottleneck instead.");
        exit(-1);
    }
}

void ResNet_init(ResNet *ResNetInstance, BlockType block_type, int num_layers[4],
        int num_classes) {
    ResNetInstance->block_type = block_type;
    ResNetInstance->inplanes = 64;
    ResNetInstance->conv1  = malloc(sizeof(ConvLayer));
    ResNetInstance->bn1    = new_bn_layer(64);
    ResNetInstance->pool1  = malloc(sizeof(ConvLayer));
    ResNetInstance->block1 = malloc(sizeof(ResNetBlock));
    ResNetInstance->block2 = malloc(sizeof(ResNetBlock));
    ResNetInstance->block3 = malloc(sizeof(ResNetBlock));
    ResNetInstance->block4 = malloc(sizeof(ResNetBlock));
    ResNetInstance->pool2  = malloc(sizeof(PoolLayer));
    ResNetInstance->fc = new_dense_layer(num_classes, 512 * (block_type == UseBasicBlock ? 1 : 4));
    ResNetInstance->output = NULL;

    ResNet_block_init(ResNetInstance->block1, ResNetInstance, 64, num_layers[0], 1);
    ResNet_block_init(ResNetInstance->block2, ResNetInstance, 128, num_layers[1], 2);
    ResNet_block_init(ResNetInstance->block3, ResNetInstance, 256, num_layers[2], 2);
    ResNet_block_init(ResNetInstance->block4, ResNetInstance, 512, num_layers[3], 1);

    conv_layer_init(ResNetInstance->conv1, 3, 64, 7, 7, 2, 2, 3);
    new_pool_layer(ResNetInstance->pool1, 3, 3, 2, 2, 0, MAXPOOL);
    new_pool_layer(ResNetInstance->pool2, 7, 7, 1, 1, 0, AVGPOOL);

    if(!LOAD_PRETRAINED_WEIGHT){
        conv_layer_rand_weight(ResNetInstance->conv1);
        batchnorm_layer_rand_weight(ResNetInstance->bn1, 64);
        dense_layer_rand_weight(ResNetInstance->fc);
    }
}

void downsample_forward(Tensor *input, Downsample *downsample) {
    conv_layer_forward(input, downsample->conv, SAVE);
    bnormLayer_forward(&(downsample->conv->out), downsample->bn, SAVE);

    downsample->output = &(downsample->conv->out);
}

void basicblock_forward(Tensor *input, BasicBlock *basicblock) {
    Tensor *residual = input;
    conv_layer_forward(input, basicblock->conv1, SAVE);
    bnormLayer_forward(&(basicblock->conv1->out), basicblock->bn1, SAVE);
    relu_forward(&(basicblock->conv1->out));

    conv_layer_forward(&(basicblock->conv1->out), basicblock->conv2, SAVE);
    bnormLayer_forward(&(basicblock->conv2->out), basicblock->bn2, SAVE);

    if (basicblock->downsample) {
        downsample_forward(input, basicblock->downsample);
        residual = basicblock->downsample->output;
    }

    for (int idx = 0; idx < tensor_len(&(basicblock->conv2->out)); idx++) {
        basicblock->conv2->out.data[idx] += residual->data[idx];
    }
    relu_forward(&(basicblock->conv2->out));

    basicblock->output = &(basicblock->conv2->out);
}

void bottleneck_forward(Tensor *input, Bottleneck *bottleneck) {
    Tensor *residual = input;
    conv_layer_forward(input, bottleneck->conv1, SAVE);
    bnormLayer_forward(&(bottleneck->conv1->out), bottleneck->bn1, SAVE);
    relu_forward(&(bottleneck->conv1->out));

    conv_layer_forward(&(bottleneck->conv1->out), bottleneck->conv2, SAVE);
    bnormLayer_forward(&(bottleneck->conv2->out), bottleneck->bn2, SAVE);
    relu_forward(&(bottleneck->conv2->out));

    conv_layer_forward(&(bottleneck->conv2->out), bottleneck->conv3, SAVE);
    bnormLayer_forward(&(bottleneck->conv3->out), bottleneck->bn3, SAVE);

    if (bottleneck->downsample) {
        downsample_forward(input, bottleneck->downsample);
        residual = bottleneck->downsample->output;
    }

    for (int idx = 0; idx < tensor_len(&(bottleneck->conv3->out)); idx++) {
        bottleneck->conv3->out.data[idx] += residual->data[idx];
    }
    relu_forward(&(bottleneck->conv3->out));

    bottleneck->output = &(bottleneck->conv3->out);
}

void resnet_block_forward(Tensor *input, ResNetBlock *block) {
    if (block->block_type == UseBasicBlock) {
        basicblock_forward(input, block->basicblocks[0]);
        for (int block_idx = 1; block_idx < block->num_blocks; block_idx++)
            basicblock_forward(
                    block->basicblocks[block_idx - 1]->output,
                    block->basicblocks[block_idx]
            );
        block->output = block->basicblocks[block->num_blocks - 1]->output;
    } else if (block->block_type == UseBottleneck) {
        bottleneck_forward(input, block->bottlenecks[0]);
        for (int block_idx = 1; block_idx < block->num_blocks; block_idx++)
            bottleneck_forward(
                    block->bottlenecks[block_idx - 1]->output,
                    block->bottlenecks[block_idx]
            );
        block->output = block->bottlenecks[block->num_blocks - 1]->output;
    }
}

void resnet_forward(Tensor *image_tensor, ResNet *resnet) {
    conv_layer_forward(image_tensor, resnet->conv1, SAVE);
    bnormLayer_forward(&(resnet->conv1->out), resnet->bn1, SAVE);
    relu_forward(&(resnet->conv1->out));
    pool_layer_forward(&(resnet->conv1->out), resnet->pool1);

    resnet_block_forward(&(resnet->pool1->out), resnet->block1);
    resnet_block_forward(resnet->block1->output, resnet->block2);
    resnet_block_forward(resnet->block2->output, resnet->block3);
    resnet_block_forward(resnet->block3->output, resnet->block4);

    pool_layer_forward(resnet->block4->output, resnet->pool2);
    dense_layer_forward(&(resnet->pool2->out), resnet->fc, SAVE);

    resnet->output = &(resnet->fc->out);
}



