#include "CNN/SqueezeNet.h"

Fire *new_fire_module(int inplanes, int squeeze_planes, int expand1x1_planes, int expand3x3_planes) {
    Fire * fire_ptr = (Fire*) malloc(sizeof(fire_ptr));

    fire_ptr->inplanes  = inplanes;
    fire_ptr->squeeze   = new_conv_layer(inplanes, squeeze_planes, 1, 1, 1, 1, 0);
    fire_ptr->expand1x1 = new_conv_layer(squeeze_planes, expand1x1_planes, 1, 1, 1, 1, 0);
    fire_ptr->expand3x3 = new_conv_layer(squeeze_planes, expand3x3_planes, 3, 3, 1, 1, 1);

    if(!LOAD_PRETRAINED_WEIGHT){
        init_conv_layer(fire_ptr->squeeze);
        init_conv_layer(fire_ptr->expand1x1);
        init_conv_layer(fire_ptr->expand3x3);
    }
    fire_ptr->output = NULL;

    return fire_ptr;
}

FeaturesSequential *new_features_sequential(SqueezeNetVersion version) {
    FeaturesSequential *features_ptr = (FeaturesSequential*) malloc(sizeof(FeaturesSequential));

    features_ptr->version = version;
    if(features_ptr->version == Version1_0){

        features_ptr->conv = new_conv_layer(3, 96, 7, 7, 2, 2, 0);
        features_ptr->fire_list[0] = new_fire_module(96,  16, 64,  64);

        if(!LOAD_PRETRAINED_WEIGHT) init_conv_layer(features_ptr->conv);

    }else if(features_ptr->version == Version1_1){

        features_ptr->conv = new_conv_layer(3, 64, 3, 3, 2, 2, 0);
        features_ptr->fire_list[0] = new_fire_module(64,  16, 64,  64);

        if(!LOAD_PRETRAINED_WEIGHT) init_conv_layer(features_ptr->conv);

    }else{
        printf("Unsupported SqueezeNet version: %d\n 1.0 or 1.1 expected.", version);
        exit(-1);
    }

    features_ptr->fire_list[1] = new_fire_module(128, 16, 64,  64);
    features_ptr->fire_list[2] = new_fire_module(128, 32, 128, 128);
    features_ptr->fire_list[3] = new_fire_module(256, 32, 128, 128);
    features_ptr->fire_list[4] = new_fire_module(256, 48, 192, 192);
    features_ptr->fire_list[5] = new_fire_module(384, 48, 192, 192);
    features_ptr->fire_list[6] = new_fire_module(384, 64, 256, 256);
    features_ptr->fire_list[7] = new_fire_module(512, 64, 256, 256);

    for(int maxpool_idx = 0; maxpool_idx < 3; maxpool_idx++){
        features_ptr->maxpool_list[maxpool_idx] = new_pool_layer(3, 3, 2, 2, 0, MAXPOOL);
    }
    features_ptr->output = NULL;

    return features_ptr;
}

ClassifierSequential *new_classifier_sequential(int num_classes) {
    ClassifierSequential *classifier_ptr = (ClassifierSequential*) malloc(sizeof(ClassifierSequential*));

    classifier_ptr->num_classes = num_classes;
    classifier_ptr->dropout     = new_dropout_layer(0.5);
    classifier_ptr->final_conv  = new_conv_layer(512, num_classes, 1, 1, 1, 1, 0);
    classifier_ptr->avgpool     = new_pool_layer(13, 13, 1, 1, 0, AVGPOOL);
    classifier_ptr->output      = NULL;

    return classifier_ptr;
}

SqueezeNet *SqueezeNet_init(SqueezeNetVersion version, int num_classes) {
    SqueezeNet *squeeze_net = (SqueezeNet*) malloc(sizeof(SqueezeNet));

    squeeze_net->version     = version;
    squeeze_net->num_classes = num_classes;
    squeeze_net->features    = new_features_sequential(version);
    squeeze_net->classifier  = new_classifier_sequential(num_classes);
    squeeze_net->output      = NULL;

    return squeeze_net;
}

void fire_forward(Tensor *input, Fire *fire) {
    convLayer_forward(input, fire->squeeze, SAVE);
    reluAct_forward(&(fire->squeeze->out));

    convLayer_forward(&(fire->squeeze->out), fire->expand1x1, SAVE);
    reluAct_forward(&(fire->expand1x1->out));
    convLayer_forward(&(fire->squeeze->out), fire->expand3x3, SAVE);
    reluAct_forward(&(fire->expand3x3->out));

    fire->output = tensor_cat(&(fire->expand1x1->out), &(fire->expand3x3->out), 3);
}

void features_forward(Tensor *input, FeaturesSequential *features) {
    convLayer_forward(input, features->conv, SAVE);
    reluAct_forward(&(features->conv->out));
    poolLayer_forward(&(features->conv->out), features->maxpool_list[0]);
    fire_forward(&(features->maxpool_list[0]->out), features->fire_list[0]);
    fire_forward(features->fire_list[0]->output, features->fire_list[1]);

    if(features->version == Version1_0){
        fire_forward(features->fire_list[1]->output, features->fire_list[2]);
        poolLayer_forward(features->fire_list[2]->output, features->maxpool_list[1]);
        fire_forward(&(features->maxpool_list[1]->out), features->fire_list[3]);
        fire_forward(features->fire_list[3]->output, features->fire_list[4]);
        fire_forward(features->fire_list[4]->output, features->fire_list[5]);
        fire_forward(features->fire_list[5]->output, features->fire_list[6]);
        poolLayer_forward(features->fire_list[6]->output, features->maxpool_list[2]);
        fire_forward(&(features->maxpool_list[2]->out), features->fire_list[7]);
    }else if(features->version == Version1_1){
        poolLayer_forward(features->fire_list[1]->output, features->maxpool_list[1]);
        fire_forward(&(features->maxpool_list[1]->out), features->fire_list[2]);
        fire_forward(features->fire_list[2]->output, features->fire_list[3]);
        poolLayer_forward(features->fire_list[3]->output, features->maxpool_list[2]);
        fire_forward(&(features->maxpool_list[2]->out), features->fire_list[4]);
        fire_forward(features->fire_list[4]->output, features->fire_list[5]);
        fire_forward(features->fire_list[5]->output, features->fire_list[6]);
        fire_forward(features->fire_list[6]->output, features->fire_list[7]);
    }else{
        fprintf(stderr, "\nWrong version of Squeeze Net\n");
    }

    features->output = features->fire_list[7]->output;
}

void classification_forward(Tensor *input, ClassifierSequential *classifier) {
    dropoutLayer_forward(input, classifier->dropout);
    convLayer_forward(input, classifier->final_conv, SAVE);
    reluAct_forward(&(classifier->final_conv->out));
    poolLayer_forward(&(classifier->final_conv->out), classifier->avgpool);

    classifier->output = &(classifier->avgpool->out);
}

void squeezenet_forward(Tensor *input, SqueezeNet *squeeze_net) {
    features_forward(input, squeeze_net->features);
    classification_forward(squeeze_net->features->output, squeeze_net->classifier);
    squeeze_net->output = squeeze_net->classifier->output;
}
