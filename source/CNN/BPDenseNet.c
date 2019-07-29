#include "CNN/BPDenseNet.h"

void bp_densenet_layer_init(BPDenseNetLayer *layer, int num_input_features,
        int bn_size, int growth_rate, float drop_rate) {
    layer->drop_rate = drop_rate;
    layer->bnorm1  = malloc(sizeof(BPBnormLayer));
    layer->conv1   = malloc(sizeof(BPConvLayer));
    layer->bnorm2  = malloc(sizeof(BPBnormLayer));
    layer->conv2   = malloc(sizeof(BPConvLayer));
    layer->dropout = malloc(sizeof(BPDropoutLayer));

    bp_bnorm_layer_init(layer->bnorm1, num_input_features);
    bp_conv_layer_init(layer->conv1, num_input_features, bn_size * growth_rate, 1, 1, 1, 1, 0);
    bp_bnorm_layer_init(layer->bnorm2, bn_size * growth_rate);
    bp_conv_layer_init(layer->conv2, bn_size * growth_rate, growth_rate, 3, 3, 1, 1, 1);
    bp_dropout_layer_init(layer->dropout, layer->drop_rate);

    if (!LOAD_PRETRAINED_WEIGHT){
        bp_bnorm_layer_rand_weight(layer->bnorm1);
        bp_conv_layer_rand_weight(layer->conv1);
        bp_bnorm_layer_rand_weight(layer->bnorm2);
        bp_conv_layer_rand_weight(layer->conv2);
    }
}

void bp_transition_init(BPTransition *transition, int num_input_faetures,
        int num_output_faetures) {
    transition->bnorm = malloc(sizeof(BPBnormLayer));
    transition->conv  = malloc(sizeof(BPConvLayer));
    transition->pool  = malloc(sizeof(BPPoolLayer));

    bp_bnorm_layer_init(transition->bnorm, num_input_faetures);
    bp_conv_layer_init(transition->conv, num_input_faetures, num_output_faetures, 1, 1, 1, 1, 0);
    bp_pool_layer_init(transition->pool, 2, 2, 2, 2, 0, AVGPOOL);

    if (!LOAD_PRETRAINED_WEIGHT){
        bp_bnorm_layer_rand_weight(transition->bnorm);
        bp_conv_layer_rand_weight(transition->conv);
    }
}

void bp_densenet_block_init(BPDenseBlock *block, int num_layers, int num_input_features,
        int bn_size, int growth_rate, float drop_rate) {
    block->num_layers = num_layers;
    block->num_input_features = num_input_features;
    block->bn_size = bn_size;
    block->growth_rate = growth_rate;
    block->drop_rate = drop_rate;
    block->densenet_layer_list = malloc(num_layers * sizeof(BPDenseNetLayer*));

    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++){
        block->densenet_layer_list[layer_idx] = malloc(sizeof(BPDenseNetLayer));
        bp_densenet_layer_init(block->densenet_layer_list[layer_idx],
                            num_input_features + layer_idx * growth_rate, bn_size, growth_rate, drop_rate);
    }
}

void bp_densenet_features_init(BPDenseNetFeatures *features, const int *block_config, int num_init_features,
        int growth_rate, int bn_size, float drop_rate) {
    features->conv0  = malloc(sizeof(BPConvLayer));
    features->bnorm0 = malloc(sizeof(BPBnormLayer));
    features->pool0  = malloc(sizeof(BPPoolLayer));
    features->block_config = block_config;
    features->dense_block_list = malloc(4 * sizeof(BPDenseBlock*));
    features->transition_list  = malloc(3 * sizeof(BPTransition*));

    int num_features = num_init_features;
    for (int i = 0; i < 4; i++){
        features->dense_block_list[i] = malloc(sizeof(BPDenseBlock));
        bp_densenet_block_init(features->dense_block_list[i], block_config[i], num_features,
                            bn_size, growth_rate, drop_rate);
        num_features += (block_config[i] * growth_rate);
        if (i != 3){
            features->transition_list[i] = malloc(sizeof(BPTransition));
            bp_transition_init(features->transition_list[i], num_features, num_features/2);
            num_features /= 2;
        }
    }

    features->bnorm5 = malloc(sizeof(BPBnormLayer));

    bp_conv_layer_init(features->conv0, 3, num_init_features, 7, 7, 2, 2, 3);
    bp_bnorm_layer_init(features->bnorm0, num_init_features);
    bp_pool_layer_init(features->pool0, 3, 3, 2, 2, 1, MAXPOOL);
    bp_bnorm_layer_init(features->bnorm5, num_features);

    if (!LOAD_PRETRAINED_WEIGHT){
        bp_conv_layer_rand_weight(features->conv0);
        bp_bnorm_layer_rand_weight(features->bnorm0);
        bp_bnorm_layer_rand_weight(features->bnorm5);
    }
    features->num_features = num_features;
}

void BPDenseNet_init(BPDenseNet *densenet, const int *block_config, int num_init_features, int growth_rate,
        int bn_size, float drop_rate, int num_classes) {
    densenet->features   = malloc(sizeof(BPDenseNetFeatures));
    densenet->classifier = malloc(sizeof(BPDenseOutputLayer));
    densenet->avgpool    = malloc(sizeof(BPPoolLayer));

    bp_densenet_features_init(densenet->features, block_config, num_init_features,
                           growth_rate, bn_size, drop_rate);
    bp_dense_output_layer_init(densenet->classifier, num_classes, densenet->features->num_features);
    bp_pool_layer_init(densenet->avgpool, 7, 7, 1, 1, 0, AVGPOOL);

    if (!LOAD_PRETRAINED_WEIGHT)
        bp_dense_output_layer_rand_weight(densenet->classifier);
}

void bp_densenet_layer_forward(Tensor *input, BPDenseNetLayer *layer) {
    Tensor tmp = bp_tensor_copy(input);
    bp_bnorm_layer_forward(input, layer->bnorm1, SAVE);
    //relu_forward(input);
    bp_conv_layer_forward(input, layer->conv1, SAVE);

    bp_bnorm_layer_forward(&(layer->conv1->out), layer->bnorm2, SAVE);
    //relu_forward(&(layer->conv1->out));
    bp_conv_layer_forward(&(layer->conv1->out), layer->conv2, SAVE);

    bp_dropout_layer_forward(&(layer->conv2->out), layer->dropout);
    if (layer->output.data != NULL)
        free(layer->output.data);
    bp_tensor_cat(&tmp, &(layer->conv2->out), &(layer->output), 3);
    bp_tensor_free(&tmp);
}

void bp_densenet_transition_forward(Tensor *input, BPTransition *transition) {
    bp_bnorm_layer_forward(input, transition->bnorm, SAVE);
    //relu_forward(input);
    bp_conv_layer_forward(input, transition->conv, SAVE);
    bp_pool_layer_forward(&(transition->conv->out), transition->pool);

    if (transition->output.data != NULL)
        free(transition->output.data);
    transition->output = bp_tensor_copy(&(transition->pool->out));
}

void bp_densenet_block_forward(Tensor *input, BPDenseBlock *block) {
    Tensor *tmp = input;
    for (int layer_idx = 0; layer_idx < block->num_layers; layer_idx++){
        bp_densenet_layer_forward(tmp, block->densenet_layer_list[layer_idx]);
        tmp = &(block->densenet_layer_list[layer_idx]->output);
    }
    if (block->output.data != NULL)
        free(block->output.data);
    block->output = bp_tensor_copy(tmp);
}

void bp_densenet_features_forward(Tensor *input, BPDenseNetFeatures *features) {
    bp_conv_layer_forward(input, features->conv0, SAVE);
    bp_bnorm_layer_forward(&(features->conv0->out), features->bnorm0, SAVE);
    //relu_forward(&(features->conv0->out));
    bp_pool_layer_forward(&(features->conv0->out), features->pool0);

    Tensor *tmp = &(features->pool0->out);
    for (int i = 0; i < 4; i++){
        bp_densenet_block_forward(tmp, features->dense_block_list[i]);
        tmp = &(features->dense_block_list[i]->output);
        if (i != 3){
            bp_densenet_transition_forward(tmp, features->transition_list[i]);
            tmp = &(features->transition_list[i]->output);
        }
    }
    bp_bnorm_layer_forward(tmp, features->bnorm5, SAVE);

    if (features->output.data != NULL)
        free(features->output.data);
    features->output = bp_tensor_copy(tmp);
}

void BPDenseNet_forward(Tensor *input, BPDenseNet *densenet) {
    bp_densenet_features_forward(input, densenet->features);
    //relu_forward(&(densenet->features->output));
    bp_pool_layer_forward(&(densenet->features->output), densenet->avgpool);
    bp_dense_output_layer_forward(&(densenet->avgpool->out), densenet->classifier, SAVE);

    densenet->output = densenet->classifier->output_arr;
}

void bp_densenet_layer_free(BPDenseNetLayer *layer){
    bp_bnorm_layer_free(layer->bnorm1);
    bp_bnorm_layer_free(layer->bnorm2);
    bp_conv_layer_free(layer->conv1);
    bp_conv_layer_free(layer->conv2);
    bp_tensor_free(&(layer->output));

    free(layer->bnorm1);
    free(layer->bnorm2);
    free(layer->conv1);
    free(layer->conv2);
    free(layer->dropout);
}

void bp_transition_free(BPTransition *transition){
    bp_bnorm_layer_free(transition->bnorm);
    bp_conv_layer_free(transition->conv);
    bp_poolLayer_free(transition->pool);
    bp_tensor_free(&(transition->output));

    free(transition->bnorm);
    free(transition->conv);
    free(transition->pool);
}

void bp_densenet_block_free(BPDenseBlock *block){
    for (int i = 0; i < block->num_layers; i++){
        bp_densenet_layer_free(block->densenet_layer_list[i]);
        free(block->densenet_layer_list[i]);
    }
    bp_tensor_free(&(block->output));

    free(block->densenet_layer_list);
}

void bp_densenet_features_free(BPDenseNetFeatures *features){
    bp_conv_layer_free(features->conv0);
    bp_bnorm_layer_free(features->bnorm0);
    bp_poolLayer_free(features->pool0);
    bp_bnorm_layer_free(features->bnorm5);

    for (int i = 0; i < 4; i++){
        bp_densenet_block_free(features->dense_block_list[i]);
        free(features->dense_block_list[i]);
        if (i != 3){
            bp_transition_free(features->transition_list[i]);
            free(features->transition_list[i]);
        }
    }

    bp_tensor_free(&(features->output));

    free(features->conv0);
    free(features->bnorm0);
    free(features->pool0);
    free(features->bnorm5);
    free(features->dense_block_list);
    free(features->transition_list);
}

void BPDenseNet_free(BPDenseNet *densenet) {
    bp_densenet_features_free(densenet->features);
    bp_dense_output_layer_free(densenet->classifier);
    bp_poolLayer_free(densenet->avgpool);
    free(densenet->output);

    free(densenet->features);
    free(densenet->classifier);
    free(densenet->avgpool);
}


