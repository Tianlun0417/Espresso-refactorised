#include "CNN/ResNet.h"


void basicblock_init(BasicBlock *basicblock, int inplanes, int planes,
                     int stride, Downsample *downsample) {
    basicblock->expansion = 1;
    basicblock->stride = stride;
    basicblock->conv1 = malloc(sizeof(ConvLayer));
    basicblock->conv2 = malloc(sizeof(ConvLayer));
    basicblock->bn1 = malloc(sizeof(bnormLayer));
    basicblock->bn2 = malloc(sizeof(bnormLayer));
    basicblock->downsample = downsample;

    conv_layer_init(basicblock->conv1, inplanes, planes, 3, 3, stride, stride, 1);
    conv_layer_init(basicblock->conv2, planes, planes, 3, 3, stride, stride, 1);
    bn_layer_init(basicblock->bn1, planes);
    bn_layer_init(basicblock->bn2, planes);

    if(!LOAD_PRETRAINED_WEIGHT){
        conv_layer_rand_weight(basicblock->conv1);
        batchnorm_layer_rand_weight(basicblock->bn1);
        conv_layer_rand_weight(basicblock->conv2);
        batchnorm_layer_rand_weight(basicblock->bn2);
    }
}

void bottleneck_init(Bottleneck *bottleneck, int inplanes, int planes,
                     int stride, Downsample *downsample) {
    bottleneck->expansion = 4;
    bottleneck->stride = stride;
    bottleneck->conv1 = malloc(sizeof(ConvLayer));
    bottleneck->conv2 = malloc(sizeof(ConvLayer));
    bottleneck->conv3 = malloc(sizeof(ConvLayer));
    bottleneck->bn1 = malloc(sizeof(bnormLayer));
    bottleneck->bn2 = malloc(sizeof(bnormLayer));
    bottleneck->bn3 = malloc(sizeof(bnormLayer));
    bottleneck->downsample = downsample;

    conv_layer_init(bottleneck->conv1, inplanes, planes, 1, 1, 1, 1, 0);
    conv_layer_init(bottleneck->conv2, planes, planes, 3, 3, stride, stride, 1);
    conv_layer_init(bottleneck->conv3, planes, planes * 4, 1, 1, 1, 1, 0);
    bn_layer_init(bottleneck->bn1, planes);
    bn_layer_init(bottleneck->bn2, planes);
    bn_layer_init(bottleneck->bn3, planes * 4);

    if(!LOAD_PRETRAINED_WEIGHT){
        conv_layer_rand_weight(bottleneck->conv1);
        batchnorm_layer_rand_weight(bottleneck->bn1);
        conv_layer_rand_weight(bottleneck->conv2);
        batchnorm_layer_rand_weight(bottleneck->bn2);
        conv_layer_rand_weight(bottleneck->conv3);
        batchnorm_layer_rand_weight(bottleneck->bn3);
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
        downsample_ptr->bn = malloc(sizeof(bnormLayer));
        bn_layer_init(downsample_ptr->bn, planes * expansion);

        if(!LOAD_PRETRAINED_WEIGHT){
            conv_layer_rand_weight(downsample_ptr->conv);
            batchnorm_layer_rand_weight(downsample_ptr->bn);
        }
    }

    if (resnet_ptr->block_type == UseBasicBlock) {
        block_ptr->bottlenecks = NULL;
        // Init the memory for basic blocks
        block_ptr->basicblocks = (BasicBlock**) malloc((num_blocks) * sizeof(BasicBlock*));
        // Init each basic block
        block_ptr->basicblocks[0] = malloc(sizeof(BasicBlock));
        basicblock_init(block_ptr->basicblocks[0], resnet_ptr->inplanes, planes, stride, downsample_ptr);
        resnet_ptr->inplanes = planes * 1;
        for (int block_idx = 1; block_idx < num_blocks; block_idx++) {
            block_ptr->basicblocks[block_idx] = malloc(sizeof(BasicBlock));
            basicblock_init(block_ptr->basicblocks[block_idx], resnet_ptr->inplanes, planes, 1, NULL);
        }
    } else if (resnet_ptr->block_type == UseBottleneck) {
        block_ptr->basicblocks = NULL;
        block_ptr->bottlenecks = (Bottleneck**) malloc((num_blocks) * sizeof(Bottleneck*));
        block_ptr->bottlenecks[0] = malloc(sizeof(Bottleneck));
        bottleneck_init(block_ptr->bottlenecks[0], resnet_ptr->inplanes, planes, stride, downsample_ptr);
        resnet_ptr->inplanes = planes * 4;
        for (int block_idx = 1; block_idx < num_blocks; block_idx++) {
            block_ptr->bottlenecks[block_idx] = malloc(sizeof(Bottleneck));
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
    ResNetInstance->bn1    = malloc(sizeof(bnormLayer));
    ResNetInstance->pool1  = malloc(sizeof(ConvLayer));
    ResNetInstance->block1 = malloc(sizeof(ResNetBlock));
    ResNetInstance->block2 = malloc(sizeof(ResNetBlock));
    ResNetInstance->block3 = malloc(sizeof(ResNetBlock));
    ResNetInstance->block4 = malloc(sizeof(ResNetBlock));
    ResNetInstance->pool2  = malloc(sizeof(PoolLayer));
    ResNetInstance->fc = malloc(sizeof(DenseLayer));

    ResNet_block_init(ResNetInstance->block1, ResNetInstance, 64, num_layers[0], 1);
    ResNet_block_init(ResNetInstance->block2, ResNetInstance, 128, num_layers[1], 2);
    ResNet_block_init(ResNetInstance->block3, ResNetInstance, 256, num_layers[2], 2);
    ResNet_block_init(ResNetInstance->block4, ResNetInstance, 512, num_layers[3], 1);

    conv_layer_init(ResNetInstance->conv1, 3, 64, 7, 7, 2, 2, 3);
    new_pool_layer(ResNetInstance->pool1, 3, 3, 2, 2, 0, MAXPOOL);
    new_pool_layer(ResNetInstance->pool2, 7, 7, 1, 1, 0, AVGPOOL);
    bn_layer_init(ResNetInstance->bn1, 64);
    dense_layer_init(ResNetInstance->fc, num_classes, 512 * (block_type == UseBasicBlock ? 1 : 4));

    if(!LOAD_PRETRAINED_WEIGHT){
        conv_layer_rand_weight(ResNetInstance->conv1);
        batchnorm_layer_rand_weight(ResNetInstance->bn1);
        dense_layer_rand_weight(ResNetInstance->fc);
    }
}

void downsample_forward(Tensor *input, Downsample *downsample) {
    conv_layer_forward(input, downsample->conv, SAVE);
    bnormLayer_forward(&(downsample->conv->out), downsample->bn, SAVE);

    if (downsample->output.data != NULL)
        free(downsample->output.data);
    downsample->output = tensor_copy(&(downsample->conv->out));
}

void basicblock_forward(Tensor *input, BasicBlock *basicblock) {
    conv_layer_forward(input, basicblock->conv1, SAVE);
    bnormLayer_forward(&(basicblock->conv1->out), basicblock->bn1, SAVE);
    relu_forward(&(basicblock->conv1->out));

    conv_layer_forward(&(basicblock->conv1->out), basicblock->conv2, SAVE);
    bnormLayer_forward(&(basicblock->conv2->out), basicblock->bn2, SAVE);

    if (basicblock->residual.data != NULL)
        free(basicblock->residual.data);

    if (basicblock->downsample) {
        downsample_forward(input, basicblock->downsample);
        basicblock->residual = tensor_copy(&(basicblock->downsample->output));
    }else{
        basicblock->residual = tensor_copy(input);
    }

    for (int idx = 0; idx < tensor_len(&(basicblock->conv2->out)); idx++) {
        basicblock->conv2->out.data[idx] += basicblock->residual.data[idx];
    }
    relu_forward(&(basicblock->conv2->out));

    if (basicblock->output.data != NULL)
        free(basicblock->output.data);
    basicblock->output = tensor_copy(&(basicblock->conv2->out));
}

void bottleneck_forward(Tensor *input, Bottleneck *bottleneck) {
    conv_layer_forward(input, bottleneck->conv1, SAVE);
    bnormLayer_forward(&(bottleneck->conv1->out), bottleneck->bn1, SAVE);
    relu_forward(&(bottleneck->conv1->out));

    conv_layer_forward(&(bottleneck->conv1->out), bottleneck->conv2, SAVE);
    bnormLayer_forward(&(bottleneck->conv2->out), bottleneck->bn2, SAVE);
    relu_forward(&(bottleneck->conv2->out));

    conv_layer_forward(&(bottleneck->conv2->out), bottleneck->conv3, SAVE);
    bnormLayer_forward(&(bottleneck->conv3->out), bottleneck->bn3, SAVE);

    if (bottleneck->residual.data != NULL)
        free(bottleneck->residual.data);

    if (bottleneck->downsample) {
        downsample_forward(input, bottleneck->downsample);
        bottleneck->residual = tensor_copy(&(bottleneck->downsample->output));
    }else{
        bottleneck->residual = tensor_copy(input);
    }

    for (int idx = 0; idx < tensor_len(&(bottleneck->conv3->out)); idx++) {
        bottleneck->conv3->out.data[idx] += bottleneck->residual.data[idx];
    }
    relu_forward(&(bottleneck->conv3->out));

    if (bottleneck->output.data != NULL)
        free(bottleneck->output.data);
    bottleneck->output = tensor_copy(&(bottleneck->conv3->out));
}

void resnet_block_forward(Tensor *input, ResNetBlock *block) {
    if (block->block_type == UseBasicBlock) {
        basicblock_forward(input, block->basicblocks[0]);
        for (int block_idx = 1; block_idx < block->num_blocks; block_idx++)
            basicblock_forward(
                    &(block->basicblocks[block_idx - 1]->output),
                    block->basicblocks[block_idx]
            );
        if (block->output.data != NULL)
            free(block->output.data);
        block->output = tensor_copy(&(block->basicblocks[block->num_blocks - 1]->output));
    } else if (block->block_type == UseBottleneck) {
        bottleneck_forward(input, block->bottlenecks[0]);
        for (int block_idx = 1; block_idx < block->num_blocks; block_idx++)
            bottleneck_forward(
                    &(block->bottlenecks[block_idx - 1]->output),
                    block->bottlenecks[block_idx]
            );
        if (block->output.data != NULL)
            free(block->output.data);
        block->output = tensor_copy(&(block->bottlenecks[block->num_blocks - 1]->output));
    }
}

void resnet_forward(Tensor *image_tensor, ResNet *resnet) {
    conv_layer_forward(image_tensor, resnet->conv1, SAVE);
    bnormLayer_forward(&(resnet->conv1->out), resnet->bn1, SAVE);
    relu_forward(&(resnet->conv1->out));
    pool_layer_forward(&(resnet->conv1->out), resnet->pool1);

    resnet_block_forward(&(resnet->pool1->out), resnet->block1);
    resnet_block_forward(&(resnet->block1->output), resnet->block2);
    resnet_block_forward(&(resnet->block2->output), resnet->block3);
    resnet_block_forward(&(resnet->block3->output), resnet->block4);

    pool_layer_forward(&(resnet->block4->output), resnet->pool2);
    dense_layer_forward(&(resnet->pool2->out), resnet->fc, SAVE);

    if (resnet->output.data != NULL)
        free(resnet->output.data);
    resnet->output = tensor_copy(&(resnet->fc->out));
}

void downsample_free(Downsample *downsample){
    conv_layer_free(downsample->conv);
    bnormLayer_free(downsample->bn);
    tensor_free(&(downsample->output));

    free(downsample->conv);
    free(downsample->bn);
}

void basicblock_free(BasicBlock *basicblock){
    conv_layer_free(basicblock->conv1);
    conv_layer_free(basicblock->conv2);
    bnormLayer_free(basicblock->bn1);
    bnormLayer_free(basicblock->bn2);
    if (basicblock->downsample != NULL) {
        downsample_free(basicblock->downsample);
        free(basicblock->downsample);
    }
    tensor_free(&(basicblock->residual));
    tensor_free(&(basicblock->output));

    free(basicblock->conv1);
    free(basicblock->conv2);
    free(basicblock->bn1);
    free(basicblock->bn2);
}

void bottleneck_free(Bottleneck *bottleneck){
    conv_layer_free(bottleneck->conv1);
    conv_layer_free(bottleneck->conv2);
    conv_layer_free(bottleneck->conv3);
    bnormLayer_free(bottleneck->bn1);
    bnormLayer_free(bottleneck->bn2);
    bnormLayer_free(bottleneck->bn3);
    if (bottleneck->downsample != NULL) {
        downsample_free(bottleneck->downsample);
        free(bottleneck->downsample);
    }
    tensor_free(&(bottleneck->residual));
    tensor_free(&(bottleneck->output));

    free(bottleneck->conv1);
    free(bottleneck->conv2);
    free(bottleneck->conv3);
    free(bottleneck->bn1);
    free(bottleneck->bn2);
    free(bottleneck->bn3);
    free(bottleneck->downsample);
}

void resnet_block_free(ResNetBlock *block){
    if (block->block_type == UseBasicBlock){
        for (int i = 0; i < block->num_blocks; i++){
            basicblock_free(block->basicblocks[i]);
            free(block->basicblocks[i]);
        }
    }else{
        for (int i = 0; i < block->num_blocks; i++){
            bottleneck_free(block->bottlenecks[i]);
            free(block->bottlenecks[i]);
        }
    }
    tensor_free(&(block->output));
    free(block->basicblocks);
    free(block->bottlenecks);
}

void ResNet_free(ResNet *resnet) {
    conv_layer_free(resnet->conv1);
    bnormLayer_free(resnet->bn1);
    poolLayer_free(resnet->pool1);
    resnet_block_free(resnet->block1);
    resnet_block_free(resnet->block2);
    resnet_block_free(resnet->block3);
    resnet_block_free(resnet->block4);
    poolLayer_free(resnet->pool2);
    denseLayer_free(resnet->fc);
    tensor_free(&(resnet->output));

    free(resnet->conv1);
    free(resnet->bn1);
    free(resnet->pool1);
    free(resnet->block1);
    free(resnet->block2);
    free(resnet->block3);
    free(resnet->block4);
    free(resnet->pool2);
    free(resnet->fc);
}



