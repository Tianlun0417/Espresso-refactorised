#include "CNN/DenseNet.h"

void densenet_layer_init(DenseNetLayer *layer, int num_input_features, int bn_size,
        int growth_rate, float drop_rate) {
    layer->drop_rate = drop_rate;
    layer->bnorm1  = malloc(sizeof(BnormLayer));
    layer->conv1   = malloc(sizeof(ConvLayer));
    layer->bnorm2  = malloc(sizeof(BnormLayer));
    layer->conv2   = malloc(sizeof(ConvLayer));
    layer->dropout = malloc(sizeof(DropoutLayer));

    bnorm_layer_init(layer->bnorm1, num_input_features);
    conv_layer_init(layer->conv1, num_input_features, bn_size * growth_rate, 1, 1, 1, 1, 0);
    bnorm_layer_init(layer->bnorm2, bn_size * growth_rate);
    conv_layer_init(layer->conv2, bn_size * growth_rate, growth_rate, 3, 3, 1, 1, 1);
    dropout_layer_init(layer->dropout, layer->drop_rate);

    if (!LOAD_PRETRAINED_WEIGHT){
        bnorm_layer_rand_weight(layer->bnorm1);
        conv_layer_rand_weight(layer->conv1);
        bnorm_layer_rand_weight(layer->bnorm2);
        conv_layer_rand_weight(layer->conv2);
    }
}

void transition_init(Transition *transition, int num_input_faetures, int num_output_faetures) {
    transition->bnorm = malloc(sizeof(BnormLayer));
    transition->conv  = malloc(sizeof(ConvLayer));
    transition->pool  = malloc(sizeof(PoolLayer));

    bnorm_layer_init(transition->bnorm, num_input_faetures);
    conv_layer_init(transition->conv, num_input_faetures, num_output_faetures, 1, 1, 1, 1, 0);
    pool_layer_init(transition->pool, 2, 2, 2, 2, 0, AVGPOOL);

    if (!LOAD_PRETRAINED_WEIGHT){
        bnorm_layer_rand_weight(transition->bnorm);
        conv_layer_rand_weight(transition->conv);
    }
}

void densenet_block_init(DenseBlock *block, int num_layers, int num_input_features,
                         int bn_size, int growth_rate, float drop_rate) {
    block->num_layers = num_layers;
    block->num_input_features = num_input_features;
    block->bn_size = bn_size;
    block->growth_rate = growth_rate;
    block->drop_rate = drop_rate;
    block->densenet_layer_list = malloc(num_layers * sizeof(DenseNetLayer*));

    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++){
        block->densenet_layer_list[layer_idx] = malloc(sizeof(DenseNetLayer));
        densenet_layer_init(block->densenet_layer_list[layer_idx],
                num_input_features + layer_idx * growth_rate, bn_size, growth_rate, drop_rate);
    }
}

void densenet_features_init(DenseNetFeatures *features, const int *block_config,
                            int num_init_features, int growth_rate, int bn_size, float drop_rate) {
    features->conv0  = malloc(sizeof(ConvLayer));
    features->bnorm0 = malloc(sizeof(BnormLayer));
    features->pool0  = malloc(sizeof(PoolLayer));
    features->block_config = block_config;
    features->dense_block_list = malloc(4 * sizeof(DenseBlock*));
    features->transition_list  = malloc(3 * sizeof(Transition*));

    int num_features = num_init_features;
    for (int i = 0; i < 4; i++){
        features->dense_block_list[i] = malloc(sizeof(DenseBlock));
        densenet_block_init(features->dense_block_list[i], block_config[i], num_features,
                            bn_size, growth_rate, drop_rate);
        num_features += (block_config[i] * growth_rate);
        if (i != 3){
            features->transition_list[i] = malloc(sizeof(Transition));
            transition_init(features->transition_list[i], num_features, num_features/2);
            num_features /= 2;
        }
    }

    features->bnorm5 = malloc(sizeof(BnormLayer));

    conv_layer_init(features->conv0, 3, num_init_features, 7, 7, 2, 2, 3);
    bnorm_layer_init(features->bnorm0, num_init_features);
    pool_layer_init(features->pool0, 3, 3, 2, 2, 1, MAXPOOL);
    bnorm_layer_init(features->bnorm5, num_features);

    if (!LOAD_PRETRAINED_WEIGHT){
        conv_layer_rand_weight(features->conv0);
        bnorm_layer_rand_weight(features->bnorm0);
        bnorm_layer_rand_weight(features->bnorm5);
    }
    features->num_features = num_features;
}

void DenseNet_init(DenseNet *densenet, const int *block_config, int num_init_features, int growth_rate,
                   int bn_size, float drop_rate, int num_classes) {
    densenet->features = malloc(sizeof(DenseNetFeatures));
    densenet->classifier = malloc(sizeof(DenseLayer));
    densenet->avgpool    = malloc(sizeof(PoolLayer));

    densenet_features_init(densenet->features, block_config, num_init_features,
                           growth_rate, bn_size, drop_rate);
    dense_layer_init(densenet->classifier, num_classes, densenet->features->num_features);
    pool_layer_init(densenet->avgpool, 7, 7, 1, 1, 0, AVGPOOL);

    if (!LOAD_PRETRAINED_WEIGHT)
        dense_layer_rand_weight(densenet->classifier);
}

void densenet_layer_forward(Tensor *input, DenseNetLayer *layer) {
    Tensor tmp = tensor_copy(input);
    bnormLayer_forward(input, layer->bnorm1, SAVE);
    relu_forward(input);
    conv_layer_forward(input, layer->conv1, SAVE);

    bnormLayer_forward(&(layer->conv1->out), layer->bnorm2, SAVE);
    relu_forward(&(layer->conv1->out));
    conv_layer_forward(&(layer->conv1->out), layer->conv2, SAVE);

    dropout_layer_forward(&(layer->conv2->out), layer->dropout);
    if (layer->output.data != NULL)
        free(layer->output.data);
    tensor_cat(&tmp, &(layer->conv2->out), &(layer->output), 3);
    tensor_free(&tmp);
}

void densenet_transition_forward(Tensor *input, Transition *transition) {
    bnormLayer_forward(input, transition->bnorm, SAVE);
    relu_forward(input);
    conv_layer_forward(input, transition->conv, SAVE);
    pool_layer_forward(&(transition->conv->out), transition->pool);

    if (transition->output.data != NULL)
        free(transition->output.data);
    transition->output = tensor_copy(&(transition->pool->out));
}

void densenet_block_forward(Tensor *input, DenseBlock *block) {
    Tensor *tmp = input;
    for (int layer_idx = 0; layer_idx < block->num_layers; layer_idx++){
        densenet_layer_forward(tmp, block->densenet_layer_list[layer_idx]);
        tmp = &(block->densenet_layer_list[layer_idx]->output);
    }
    if (block->output.data != NULL)
        free(block->output.data);
    block->output = tensor_copy(tmp);
}

void densenet_features_forward(Tensor *input, DenseNetFeatures *features) {
    conv_layer_forward(input, features->conv0, SAVE);
    bnormLayer_forward(&(features->conv0->out), features->bnorm0, SAVE);
    relu_forward(&(features->conv0->out));
    pool_layer_forward(&(features->conv0->out), features->pool0);

    Tensor *tmp = &(features->pool0->out);
    for (int i = 0; i < 4; i++){
        densenet_block_forward(tmp, features->dense_block_list[i]);
        tmp = &(features->dense_block_list[i]->output);
        if (i != 3){
            densenet_transition_forward(tmp, features->transition_list[i]);
            tmp = &(features->transition_list[i]->output);
        }
    }
    bnormLayer_forward(tmp, features->bnorm5, SAVE);

    if (features->output.data != NULL)
        free(features->output.data);
    features->output = tensor_copy(tmp);
}

void DenseNet_forward(Tensor *input, DenseNet *densenet) {
    densenet_features_forward(input, densenet->features);
    relu_forward(&(densenet->features->output));
    pool_layer_forward(&(densenet->features->output), densenet->avgpool);
    dense_layer_forward(&(densenet->avgpool->out), densenet->classifier, SAVE);

    if (densenet->output.data != NULL)
        free(densenet->output.data);
    densenet->output = tensor_copy(&(densenet->classifier->out));
}

void densenet_layer_free(DenseNetLayer *layer){
    bnormLayer_free(layer->bnorm1);
    bnormLayer_free(layer->bnorm2);
    conv_layer_free(layer->conv1);
    conv_layer_free(layer->conv2);
    tensor_free(&(layer->output));

    free(layer->bnorm1);
    free(layer->bnorm2);
    free(layer->conv1);
    free(layer->conv2);
    free(layer->dropout);
}

void transition_free(Transition *transition){
    bnormLayer_free(transition->bnorm);
    conv_layer_free(transition->conv);
    poolLayer_free(transition->pool);
    tensor_free(&(transition->output));

    free(transition->bnorm);
    free(transition->conv);
    free(transition->pool);
}

void densenet_block_free(DenseBlock *block){
    for (int i = 0; i < block->num_layers; i++){
        densenet_layer_free(block->densenet_layer_list[i]);
        free(block->densenet_layer_list[i]);
    }
    tensor_free(&(block->output));

    free(block->densenet_layer_list);
}

void densenet_features_free(DenseNetFeatures *features){
    conv_layer_free(features->conv0);
    bnormLayer_free(features->bnorm0);
    poolLayer_free(features->pool0);
    bnormLayer_free(features->bnorm5);

    for (int i = 0; i < 4; i++){
        densenet_block_free(features->dense_block_list[i]);
        free(features->dense_block_list[i]);
        if (i != 3){
            transition_free(features->transition_list[i]);
            free(features->transition_list[i]);
        }
    }

    tensor_free(&(features->output));

    free(features->conv0);
    free(features->bnorm0);
    free(features->pool0);
    free(features->bnorm5);
    free(features->dense_block_list);
    free(features->transition_list);
}

void DenseNet_free(DenseNet *densenet) {
    densenet_features_free(densenet->features);
    denseLayer_free(densenet->classifier);
    poolLayer_free(densenet->avgpool);
    tensor_free(&(densenet->output));

    free(densenet->features);
    free(densenet->classifier);
    free(densenet->avgpool);
}

