#include "CNN/SqueezeNet.h"


void fire_module_init(Fire *fire_ptr, int inplanes, int squeeze_planes, int expand1x1_planes, int expand3x3_planes) {
    fire_ptr->output    = malloc(sizeof(Tensor));
    fire_ptr->inplanes  = inplanes;
    fire_ptr->squeeze   = malloc(sizeof(ConvLayer));
    fire_ptr->expand1x1 = malloc(sizeof(ConvLayer));
    fire_ptr->expand3x3 = malloc(sizeof(ConvLayer));

    conv_layer_init(fire_ptr->squeeze, inplanes, squeeze_planes, 1, 1, 1, 1, 0);
    conv_layer_init(fire_ptr->expand1x1, squeeze_planes, expand1x1_planes, 1, 1, 1, 1, 0);
    conv_layer_init(fire_ptr->expand3x3, squeeze_planes, expand3x3_planes, 3, 3, 1, 1, 1);

    if(!LOAD_PRETRAINED_WEIGHT){
        conv_layer_rand_weight(fire_ptr->squeeze);
        conv_layer_rand_weight(fire_ptr->expand1x1);
        conv_layer_rand_weight(fire_ptr->expand3x3);
    }
}

void features_sequential_init(FeaturesSequential *features_ptr, SqueezeNetVersion version) {
    features_ptr->version = version;
    if(features_ptr->version == Version1_0){

        features_ptr->conv = malloc(sizeof(ConvLayer));
        conv_layer_init(features_ptr->conv, 3, 96, 7, 7, 2, 2, 0);
        features_ptr->fire_list[0] = malloc(sizeof(Fire));
        fire_module_init(features_ptr->fire_list[0], 96, 16, 64, 64);

        if(!LOAD_PRETRAINED_WEIGHT) conv_layer_rand_weight(features_ptr->conv);

    }else if(features_ptr->version == Version1_1){

        features_ptr->conv = (ConvLayer*) malloc(sizeof(ConvLayer));
        conv_layer_init(features_ptr->conv, 3, 64, 3, 3, 2, 2, 0);
        features_ptr->fire_list[0] = malloc(sizeof(Fire));
        fire_module_init(features_ptr->fire_list[0], 64, 16, 64, 64);

        if(!LOAD_PRETRAINED_WEIGHT) conv_layer_rand_weight(features_ptr->conv);

    }else{
        printf("Unsupported SqueezeNet version: %d\n 1.0 or 1.1 expected.", version);
        exit(-1);
    }

    features_ptr->fire_list[1] = malloc(sizeof(Fire));
    features_ptr->fire_list[2] = malloc(sizeof(Fire));
    features_ptr->fire_list[3] = malloc(sizeof(Fire));
    features_ptr->fire_list[4] = malloc(sizeof(Fire));
    features_ptr->fire_list[5] = malloc(sizeof(Fire));
    features_ptr->fire_list[6] = malloc(sizeof(Fire));
    features_ptr->fire_list[7] = malloc(sizeof(Fire));

    fire_module_init(features_ptr->fire_list[1], 128, 16, 64, 64);
    fire_module_init(features_ptr->fire_list[2], 128, 32, 128, 128);
    fire_module_init(features_ptr->fire_list[3], 256, 32, 128, 128);
    fire_module_init(features_ptr->fire_list[4], 256, 48, 192, 192);
    fire_module_init(features_ptr->fire_list[5], 384, 48, 192, 192);
    fire_module_init(features_ptr->fire_list[6], 384, 64, 256, 256);
    fire_module_init(features_ptr->fire_list[7], 512, 64, 256, 256);

    for(int maxpool_idx = 0; maxpool_idx < 3; maxpool_idx++){
        features_ptr->maxpool_list[maxpool_idx] = malloc(sizeof(PoolLayer));
        new_pool_layer(features_ptr->maxpool_list[maxpool_idx], 3, 3, 2, 2, 0, MAXPOOL);
    }
}

void classifier_sequential_init(ClassifierSequential *classifier_ptr, int num_classes) {
    classifier_ptr->num_classes = num_classes;
    classifier_ptr->dropout     = new_dropout_layer(0.5);
    classifier_ptr->final_conv  = malloc(sizeof(ConvLayer));
    classifier_ptr->avgpool     = malloc(sizeof(PoolLayer));
//    classifier_ptr->output      = malloc(sizeof(Tensor));

    new_pool_layer(classifier_ptr->avgpool, 13, 13, 1, 1, 0, AVGPOOL);
    conv_layer_init(classifier_ptr->final_conv, 512, num_classes, 1, 1, 1, 1, 0);

    if(!LOAD_PRETRAINED_WEIGHT) conv_layer_rand_weight(classifier_ptr->final_conv);
}

void SqueezeNet_init(SqueezeNet * squeeze_net_ptr, SqueezeNetVersion version, int num_classes) {
    squeeze_net_ptr->version     = version;
    squeeze_net_ptr->num_classes = num_classes;
    squeeze_net_ptr->features    = malloc(sizeof(FeaturesSequential));
    squeeze_net_ptr->classifier  = malloc(sizeof(ClassifierSequential));
//    squeeze_net_ptr->output      = malloc(sizeof(Tensor));

    features_sequential_init(squeeze_net_ptr->features, version);
    classifier_sequential_init(squeeze_net_ptr->classifier, num_classes);
}

void fire_forward(Tensor *input, Fire *fire) {
    conv_layer_forward(input, fire->squeeze, SAVE);
    relu_forward(&(fire->squeeze->out));

    conv_layer_forward(&(fire->squeeze->out), fire->expand1x1, SAVE);
    relu_forward(&(fire->expand1x1->out));
    conv_layer_forward(&(fire->squeeze->out), fire->expand3x3, SAVE);
    relu_forward(&(fire->expand3x3->out));

    tensor_cat(&(fire->expand1x1->out), &(fire->expand3x3->out), fire->output, 3);
}

void features_forward(Tensor *input, FeaturesSequential *features) {
    conv_layer_forward(input, features->conv, SAVE);
    relu_forward(&(features->conv->out));
    pool_layer_forward(&(features->conv->out), features->maxpool_list[0]);
    fire_forward(&(features->maxpool_list[0]->out), features->fire_list[0]);
    fire_forward(features->fire_list[0]->output, features->fire_list[1]);

    if(features->version == Version1_0){
        fire_forward(features->fire_list[1]->output, features->fire_list[2]);
        pool_layer_forward(features->fire_list[2]->output, features->maxpool_list[1]);
        fire_forward(&(features->maxpool_list[1]->out), features->fire_list[3]);
        fire_forward(features->fire_list[3]->output, features->fire_list[4]);
        fire_forward(features->fire_list[4]->output, features->fire_list[5]);
        fire_forward(features->fire_list[5]->output, features->fire_list[6]);
        pool_layer_forward(features->fire_list[6]->output, features->maxpool_list[2]);
        fire_forward(&(features->maxpool_list[2]->out), features->fire_list[7]);
    }else if(features->version == Version1_1){
        pool_layer_forward(features->fire_list[1]->output, features->maxpool_list[1]);
        fire_forward(&(features->maxpool_list[1]->out), features->fire_list[2]);
        fire_forward(features->fire_list[2]->output, features->fire_list[3]);
        pool_layer_forward(features->fire_list[3]->output, features->maxpool_list[2]);
        fire_forward(&(features->maxpool_list[2]->out), features->fire_list[4]);
        fire_forward(features->fire_list[4]->output, features->fire_list[5]);
        fire_forward(features->fire_list[5]->output, features->fire_list[6]);
        fire_forward(features->fire_list[6]->output, features->fire_list[7]);
    }else{
        fprintf(stderr, "\nWrong version of Squeeze Net\n");
    }

    features->output = tensor_copy(features->fire_list[7]->output);
}

void classification_forward(Tensor *input, ClassifierSequential *classifier) {
    //dropout_layer_forward(input, classifier->dropout);
    conv_layer_forward(input, classifier->final_conv, SAVE);
    relu_forward(&(classifier->final_conv->out));
    pool_layer_forward(&(classifier->final_conv->out), classifier->avgpool);

    classifier->output = tensor_copy(&(classifier->avgpool->out));
}

void squeezenet_forward(Tensor *input, SqueezeNet *squeeze_net) {
    features_forward(input, squeeze_net->features);
    classification_forward(&(squeeze_net->features->output), squeeze_net->classifier);
    squeeze_net->output = squeeze_net->classifier->output;
}

void fire_module_free(Fire *fire_ptr){
    conv_layer_free(fire_ptr->squeeze);
    conv_layer_free(fire_ptr->expand1x1);
    conv_layer_free(fire_ptr->expand3x3);
    tensor_free(fire_ptr->output);

    free(fire_ptr->squeeze);
    free(fire_ptr->expand1x1);
    free(fire_ptr->expand3x3);
    free(fire_ptr->output);
}

void features_free(FeaturesSequential *features_ptr){
    conv_layer_free(features_ptr->conv);
    free(features_ptr->conv);
    for (int i = 0; i < 8; i++){
        fire_module_free(features_ptr->fire_list[i]);
        free(features_ptr->fire_list[i]);
    }

    for (int i = 0; i < 3; i++){
        poolLayer_free(features_ptr->maxpool_list[i]);
        free(features_ptr->maxpool_list[i]);
    }
    tensor_free(&(features_ptr->output));
}

void classifier_free(ClassifierSequential *classifier_ptr){
    free(classifier_ptr->dropout);
    conv_layer_free(classifier_ptr->final_conv);
    free(classifier_ptr->final_conv);
    poolLayer_free(classifier_ptr->avgpool);
    free(classifier_ptr->avgpool);
    tensor_free(&(classifier_ptr->output));
}

void Squeezenet_free(SqueezeNet *squeeze_net) {
    features_free(squeeze_net->features);
    free(squeeze_net->features);
    classifier_free(squeeze_net->classifier);
    free(squeeze_net->classifier);
    tensor_free(&(squeeze_net->output));
}
