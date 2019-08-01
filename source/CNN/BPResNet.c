#include "CNN/BPResNet.h"

void bp_basicblock_init(BPBasicBlock *basicblock, int inplanes, int planes,
        int stride, BPDownsample *downsample) {
    basicblock->expansion = 1;
    basicblock->stride = stride;
    basicblock->conv1 = malloc(sizeof(BPConvLayer));
    basicblock->conv2 = malloc(sizeof(BPConvLayer));
    basicblock->bn1 = malloc(sizeof(BPBnormLayer));
    basicblock->bn2 = malloc(sizeof(BPBnormLayer));
    basicblock->downsample = downsample;

    bp_conv_layer_init(basicblock->conv1, inplanes, planes, 3, 3, stride, stride, 1);
    bp_conv_layer_init(basicblock->conv2, planes, planes, 3, 3, stride, stride, 1);
    bp_bnorm_layer_init(basicblock->bn1, planes);
    bp_bnorm_layer_init(basicblock->bn2, planes);

    if(!LOAD_PRETRAINED_WEIGHT){
        bp_conv_layer_rand_weight(basicblock->conv1);
        bp_bnorm_layer_rand_weight(basicblock->bn1);
        bp_conv_layer_rand_weight(basicblock->conv2);
        bp_bnorm_layer_rand_weight(basicblock->bn2);
    }
}

void bp_bottleneck_init(BPBottleneck *bottleneck, int inplanes, int planes,
        int stride, BPDownsample *downsample) {
    bottleneck->expansion = 4;
    bottleneck->stride = stride;
    bottleneck->conv1 = malloc(sizeof(BPConvLayer));
    bottleneck->conv2 = malloc(sizeof(BPConvLayer));
    bottleneck->conv3 = malloc(sizeof(BPConvLayer));
    bottleneck->bn1 = malloc(sizeof(BPBnormLayer));
    bottleneck->bn2 = malloc(sizeof(BPBnormLayer));
    bottleneck->bn3 = malloc(sizeof(BPBnormLayer));
    bottleneck->downsample = downsample;

    bp_conv_layer_init(bottleneck->conv1, inplanes, planes, 1, 1, 1, 1, 0);
    bp_conv_layer_init(bottleneck->conv2, planes, planes, 3, 3, stride, stride, 1);
    bp_conv_layer_init(bottleneck->conv3, planes, planes * 4, 1, 1, 1, 1, 0);
    bp_bnorm_layer_init(bottleneck->bn1, planes);
    bp_bnorm_layer_init(bottleneck->bn2, planes);
    bp_bnorm_layer_init(bottleneck->bn3, planes * 4);

    if(!LOAD_PRETRAINED_WEIGHT){
        bp_conv_layer_rand_weight(bottleneck->conv1);
        bp_bnorm_layer_rand_weight(bottleneck->bn1);
        bp_conv_layer_rand_weight(bottleneck->conv2);
        bp_bnorm_layer_rand_weight(bottleneck->bn2);
        bp_conv_layer_rand_weight(bottleneck->conv3);
        bp_bnorm_layer_rand_weight(bottleneck->bn3);
    }
}

void bp_resnet_block_init(BPResNetBlock *block_ptr, BPResNet *resnet_ptr, int planes,
        int num_blocks, int stride) {
    block_ptr->num_blocks = num_blocks;
    block_ptr->block_type = resnet_ptr->block_type;

    BPDownsample *downsample_ptr = NULL;
    int expansion = resnet_ptr->block_type == UseBasicBlock ? 1 : 4;

    if (stride != 1 || resnet_ptr->inplanes != planes * expansion) {
        downsample_ptr = malloc(sizeof(BPDownsample));
        downsample_ptr->conv = malloc(sizeof(BPConvLayer));
        bp_conv_layer_init(downsample_ptr->conv, resnet_ptr->inplanes, planes, 1, 1, stride, stride, 0);
        downsample_ptr->bn = malloc(sizeof(BPBnormLayer));
        bp_bnorm_layer_init(downsample_ptr->bn, planes * expansion);

        if(!LOAD_PRETRAINED_WEIGHT){
            bp_conv_layer_rand_weight(downsample_ptr->conv);
            bp_bnorm_layer_rand_weight(downsample_ptr->bn);
        }
    }

    if (resnet_ptr->block_type == UseBasicBlock) {
        block_ptr->bottlenecks = NULL;
        // Init the memory for basic blocks
        block_ptr->basicblocks = (BPBasicBlock**) malloc((num_blocks) * sizeof(BPBasicBlock*));
        // Init each basic block
        block_ptr->basicblocks[0] = malloc(sizeof(BPBasicBlock));
        bp_basicblock_init(block_ptr->basicblocks[0], resnet_ptr->inplanes, planes, stride, downsample_ptr);
        resnet_ptr->inplanes = planes * 1;
        for (int block_idx = 1; block_idx < num_blocks; block_idx++) {
            block_ptr->basicblocks[block_idx] = malloc(sizeof(BPBasicBlock));
            bp_basicblock_init(block_ptr->basicblocks[block_idx], resnet_ptr->inplanes, planes, 1, NULL);
        }
    } else if (resnet_ptr->block_type == UseBottleneck) {
        block_ptr->basicblocks = NULL;
        block_ptr->bottlenecks = (BPBottleneck**) malloc((num_blocks) * sizeof(BPBottleneck*));
        block_ptr->bottlenecks[0] = malloc(sizeof(BPBottleneck));
        bp_bottleneck_init(block_ptr->bottlenecks[0], resnet_ptr->inplanes, planes, stride, downsample_ptr);
        resnet_ptr->inplanes = planes * 4;
        for (int block_idx = 1; block_idx < num_blocks; block_idx++) {
            block_ptr->bottlenecks[block_idx] = malloc(sizeof(BPBottleneck));
            bp_bottleneck_init(block_ptr->bottlenecks[block_idx], resnet_ptr->inplanes, planes, 1, NULL);
        }
    } else {
        puts("Non-supported block type! Use BasicBlock or Bottleneck instead.");
        exit(-1);
    }
}

void BPResNet_init(BPResNet *ResNetInstance, BPBlockType block_type, int *num_layers,
        int num_classes) {
    ResNetInstance->block_type = block_type;
    ResNetInstance->inplanes = 64;
    ResNetInstance->conv1  = malloc(sizeof(BPConvLayer));
    ResNetInstance->bn1    = malloc(sizeof(BPBnormLayer));
    ResNetInstance->pool1  = malloc(sizeof(BPConvLayer));
    ResNetInstance->block1 = malloc(sizeof(BPResNetBlock));
    ResNetInstance->block2 = malloc(sizeof(BPResNetBlock));
    ResNetInstance->block3 = malloc(sizeof(BPResNetBlock));
    ResNetInstance->block4 = malloc(sizeof(BPResNetBlock));
    ResNetInstance->pool2  = malloc(sizeof(BPPoolLayer));
    ResNetInstance->fc = malloc(sizeof(BPDenseLayer));

    bp_resnet_block_init(ResNetInstance->block1, ResNetInstance, 64, num_layers[0], 1);
    bp_resnet_block_init(ResNetInstance->block2, ResNetInstance, 128, num_layers[1], 2);
    bp_resnet_block_init(ResNetInstance->block3, ResNetInstance, 256, num_layers[2], 2);
    bp_resnet_block_init(ResNetInstance->block4, ResNetInstance, 512, num_layers[3], 1);

    bp_conv_layer_init(ResNetInstance->conv1, 3, 64, 7, 7, 2, 2, 3);
    bp_pool_layer_init(ResNetInstance->pool1, 3, 3, 2, 2, 0, MAXPOOL);
    bp_pool_layer_init(ResNetInstance->pool2, 7, 7, 1, 1, 0, AVGPOOL);
    bp_bnorm_layer_init(ResNetInstance->bn1, 64);
    bp_dense_output_layer_init(ResNetInstance->fc, num_classes, 512 * (block_type == UseBasicBlock ? 1 : 4));

    if(!LOAD_PRETRAINED_WEIGHT){
        bp_conv_layer_rand_weight(ResNetInstance->conv1);
        bp_bnorm_layer_rand_weight(ResNetInstance->bn1);
        bp_dense_output_layer_rand_weight(ResNetInstance->fc);
    }
}

void bp_downsample_forward(Tensor *input, BPDownsample *downsample) {
    bp_conv_layer_forward(input, downsample->conv, SAVE);
    bp_bnorm_layer_forward(&(downsample->conv->out), downsample->bn, SAVE);

//    if (downsample->output.data != NULL)
//        free(downsample->output.data);
    downsample->output = bp_tensor_copy(&(downsample->conv->out));
}

void bp_basicblock_forward(Tensor *input, BPBasicBlock *basicblock) {
    bp_conv_layer_forward(input, basicblock->conv1, SAVE);
    bp_bnorm_layer_forward(&(basicblock->conv1->out), basicblock->bn1, SAVE);
    //relu_forward(&(basicblock->conv1->out));

    bp_conv_layer_forward(&(basicblock->conv1->out), basicblock->conv2, SAVE);
    bp_bnorm_layer_forward(&(basicblock->conv2->out), basicblock->bn2, SAVE);

//    if (basicblock->residual.data != NULL)
//        free(basicblock->residual.data);

    if (basicblock->downsample) {
        bp_downsample_forward(input, basicblock->downsample);
        basicblock->residual = bp_tensor_copy(&(basicblock->downsample->output));
    }else{
        basicblock->residual = bp_tensor_copy(input);
    }

    for (int idx = 0; idx < basicblock->conv2->out.packed_len; idx++)
        basicblock->conv2->out.data[idx] =
                basicblock->conv2->out.data[idx] | basicblock->residual.data[idx];

    //relu_forward(&(basicblock->conv2->out));

//    if (basicblock->output.data != NULL)
//        free(basicblock->output.data);
    basicblock->output = bp_tensor_copy(&(basicblock->conv2->out));
}

void bp_bottleneck_forward(Tensor *input, BPBottleneck *bottleneck) {
    bp_conv_layer_forward(input, bottleneck->conv1, SAVE);
    bp_bnorm_layer_forward(&(bottleneck->conv1->out), bottleneck->bn1, SAVE);
    //relu_forward(&(bottleneck->conv1->out));

    bp_conv_layer_forward(&(bottleneck->conv1->out), bottleneck->conv2, SAVE);
    bp_bnorm_layer_forward(&(bottleneck->conv2->out), bottleneck->bn2, SAVE);
    //relu_forward(&(bottleneck->conv2->out));

    bp_conv_layer_forward(&(bottleneck->conv2->out), bottleneck->conv3, SAVE);
    bp_bnorm_layer_forward(&(bottleneck->conv3->out), bottleneck->bn3, SAVE);

    if (bottleneck->residual.data != NULL)
        free(bottleneck->residual.data);

    if (bottleneck->downsample) {
        bp_downsample_forward(input, bottleneck->downsample);
        bottleneck->residual = bp_tensor_copy(&(bottleneck->downsample->output));
    }else{
        bottleneck->residual = bp_tensor_copy(input);
    }

    for (int idx = 0; idx < bottleneck->conv3->out.packed_len; idx++) {
        bottleneck->conv3->out.data[idx] =
                bottleneck->conv3->out.data[idx] | bottleneck->residual.data[idx];
    }
    //relu_forward(&(bottleneck->conv3->out));

    if (bottleneck->output.data != NULL)
        free(bottleneck->output.data);
    bottleneck->output = bp_tensor_copy(&(bottleneck->conv3->out));
}

void bp_resnet_block_forward(Tensor *input, BPResNetBlock *block) {
    if (block->block_type == UseBasicBlock) {
        bp_basicblock_forward(input, block->basicblocks[0]);
        for (int block_idx = 1; block_idx < block->num_blocks; block_idx++)
            bp_basicblock_forward(
                    &(block->basicblocks[block_idx - 1]->output),
                    block->basicblocks[block_idx]
            );
//        if (block->output.data != NULL)
//            free(block->output.data);
        block->output = bp_tensor_copy(&(block->basicblocks[block->num_blocks - 1]->output));
    } else if (block->block_type == UseBottleneck) {
        bp_bottleneck_forward(input, block->bottlenecks[0]);
        for (int block_idx = 1; block_idx < block->num_blocks; block_idx++)
            bp_bottleneck_forward(
                    &(block->bottlenecks[block_idx - 1]->output),
                    block->bottlenecks[block_idx]
            );
        if (block->output.data != NULL)
            free(block->output.data);
        block->output = bp_tensor_copy(&(block->bottlenecks[block->num_blocks - 1]->output));
    }
}

void BPResNet_forward(Tensor *image_tensor, BPResNet *resnet) {
    bp_conv_layer_forward(image_tensor, resnet->conv1, SAVE);
    bp_bnorm_layer_forward(&(resnet->conv1->out), resnet->bn1, SAVE);
    //relu_forward(&(resnet->conv1->out));
    bp_pool_layer_forward(&(resnet->conv1->out), resnet->pool1);

    bp_resnet_block_forward(&(resnet->pool1->out), resnet->block1);
    bp_resnet_block_forward(&(resnet->block1->output), resnet->block2);
    bp_resnet_block_forward(&(resnet->block2->output), resnet->block3);
    bp_resnet_block_forward(&(resnet->block3->output), resnet->block4);

    bp_pool_layer_forward(&(resnet->block4->output), resnet->pool2);
    bp_dense_output_layer_forward(&(resnet->pool2->out), resnet->fc, SAVE);

//    if (resnet->output != NULL)
//        free(resnet->output);
    resnet->output = resnet->fc->output_arr;
}

void bp_downsample_free(BPDownsample *downsample){
    bp_conv_layer_free(downsample->conv);
    bp_bnorm_layer_free(downsample->bn);
    bp_tensor_free(&(downsample->output));

    free(downsample->conv);
    free(downsample->bn);
}

void bp_basicblock_free(BPBasicBlock *basicblock){
    bp_conv_layer_free(basicblock->conv1);
    bp_conv_layer_free(basicblock->conv2);
    bp_bnorm_layer_free(basicblock->bn1);
    bp_bnorm_layer_free(basicblock->bn2);
    if (basicblock->downsample != NULL) {
        bp_downsample_free(basicblock->downsample);
        free(basicblock->downsample);
    }
    bp_tensor_free(&(basicblock->residual));
    bp_tensor_free(&(basicblock->output));

    free(basicblock->conv1);
    free(basicblock->conv2);
    free(basicblock->bn1);
    free(basicblock->bn2);
}

void bp_bottleneck_free(BPBottleneck *bottleneck){
    bp_conv_layer_free(bottleneck->conv1);
    bp_conv_layer_free(bottleneck->conv2);
    bp_conv_layer_free(bottleneck->conv3);
    bp_bnorm_layer_free(bottleneck->bn1);
    bp_bnorm_layer_free(bottleneck->bn2);
    bp_bnorm_layer_free(bottleneck->bn3);
    if (bottleneck->downsample != NULL) {
        bp_downsample_free(bottleneck->downsample);
        free(bottleneck->downsample);
    }
    bp_tensor_free(&(bottleneck->residual));
    bp_tensor_free(&(bottleneck->output));

    free(bottleneck->conv1);
    free(bottleneck->conv2);
    free(bottleneck->conv3);
    free(bottleneck->bn1);
    free(bottleneck->bn2);
    free(bottleneck->bn3);
    free(bottleneck->downsample);
}

void bp_resnet_block_free(BPResNetBlock *block){
    if (block->block_type == UseBasicBlock){
        for (int i = 0; i < block->num_blocks; i++){
            bp_basicblock_free(block->basicblocks[i]);
            free(block->basicblocks[i]);
        }
    }else{
        for (int i = 0; i < block->num_blocks; i++){
            bp_bottleneck_free(block->bottlenecks[i]);
            free(block->bottlenecks[i]);
        }
    }
    bp_tensor_free(&(block->output));
    free(block->basicblocks);
    free(block->bottlenecks);
}

void BPResNet_free(BPResNet *resnet) {
    bp_conv_layer_free(resnet->conv1);
    bp_bnorm_layer_free(resnet->bn1);
    bp_pool_layer_free(resnet->pool1);
    bp_resnet_block_free(resnet->block1);
    bp_resnet_block_free(resnet->block2);
    bp_resnet_block_free(resnet->block3);
    bp_resnet_block_free(resnet->block4);
    bp_pool_layer_free(resnet->pool2);

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

