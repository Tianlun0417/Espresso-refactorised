#include "CNN/BPSqueezeNet.h"

void
bp_fire_module_init(BPFire *fire_ptr, int inplanes, int squeeze_planes,
        int expand1x1_planes, int expand3x3_planes) {
    fire_ptr->output    = malloc(sizeof(Tensor));
    fire_ptr->inplanes  = inplanes;
    fire_ptr->squeeze   = malloc(sizeof(BPConvLayer));
    fire_ptr->expand1x1 = malloc(sizeof(BPConvLayer));
    fire_ptr->expand3x3 = malloc(sizeof(BPConvLayer));

    bp_conv_layer_init(fire_ptr->squeeze, inplanes, squeeze_planes, 1, 1, 1, 1, 0);
    bp_conv_layer_init(fire_ptr->expand1x1, squeeze_planes, expand1x1_planes, 1, 1, 1, 1, 0);
    bp_conv_layer_init(fire_ptr->expand3x3, squeeze_planes, expand3x3_planes, 3, 3, 1, 1, 1);

    if(!LOAD_PRETRAINED_WEIGHT){
        bp_conv_layer_rand_weight(fire_ptr->squeeze);
        bp_conv_layer_rand_weight(fire_ptr->expand1x1);
        bp_conv_layer_rand_weight(fire_ptr->expand3x3);
    }
}

void bp_features_sequential_init(BPFeaturesSequential *features_ptr,
        BPSqueezeNetVersion version) {
    features_ptr->version = version;
    if(features_ptr->version == Version1_0){

        features_ptr->conv = malloc(sizeof(BPConvLayer));
        bp_conv_layer_init(features_ptr->conv, 3, 96, 7, 7, 2, 2, 0);
        features_ptr->fire_list[0] = malloc(sizeof(BPFire));
        bp_fire_module_init(features_ptr->fire_list[0], 96, 16, 64, 64);

        if(!LOAD_PRETRAINED_WEIGHT) bp_conv_layer_rand_weight(features_ptr->conv);

    }else if(features_ptr->version == Version1_1){

        features_ptr->conv = (BPConvLayer*) malloc(sizeof(BPConvLayer));
        bp_conv_layer_init(features_ptr->conv, 3, 64, 3, 3, 2, 2, 0);
        features_ptr->fire_list[0] = malloc(sizeof(BPFire));
        bp_fire_module_init(features_ptr->fire_list[0], 64, 16, 64, 64);

        if(!LOAD_PRETRAINED_WEIGHT) bp_conv_layer_rand_weight(features_ptr->conv);

    }else{
        printf("Unsupported SqueezeNet version: %d\n 1.0 or 1.1 expected.", version);
        exit(-1);
    }

    features_ptr->fire_list[1] = malloc(sizeof(BPFire));
    features_ptr->fire_list[2] = malloc(sizeof(BPFire));
    features_ptr->fire_list[3] = malloc(sizeof(BPFire));
    features_ptr->fire_list[4] = malloc(sizeof(BPFire));
    features_ptr->fire_list[5] = malloc(sizeof(BPFire));
    features_ptr->fire_list[6] = malloc(sizeof(BPFire));
    features_ptr->fire_list[7] = malloc(sizeof(BPFire));

    bp_fire_module_init(features_ptr->fire_list[1], 128, 16, 64, 64);
    bp_fire_module_init(features_ptr->fire_list[2], 128, 32, 128, 128);
    bp_fire_module_init(features_ptr->fire_list[3], 256, 32, 128, 128);
    bp_fire_module_init(features_ptr->fire_list[4], 256, 48, 192, 192);
    bp_fire_module_init(features_ptr->fire_list[5], 384, 48, 192, 192);
    bp_fire_module_init(features_ptr->fire_list[6], 384, 64, 256, 256);
    bp_fire_module_init(features_ptr->fire_list[7], 512, 64, 256, 256);

    for(int maxpool_idx = 0; maxpool_idx < 3; maxpool_idx++){
        features_ptr->maxpool_list[maxpool_idx] = malloc(sizeof(BPPoolLayer));
        bp_pool_layer_init(features_ptr->maxpool_list[maxpool_idx], 3, 3, 2, 2, 0, BPMAXPOOL);
    }
}

void bp_classifier_sequential_init(BPClassifierSequential *classifier_ptr,
        int num_classes) {
    classifier_ptr->num_classes = num_classes;
    classifier_ptr->dropout     = malloc(sizeof(BPDropoutLayer));
    classifier_ptr->final_conv  = malloc(sizeof(BPConvLayer));
    classifier_ptr->avgpool     = malloc(sizeof(BPPoolLayer));
//    classifier_ptr->output      = malloc(sizeof(Tensor));

    bp_pool_layer_init(classifier_ptr->avgpool, 13, 13, 1, 1, 0, BPAVGPOOL);
    bp_conv_layer_init(classifier_ptr->final_conv, 512, num_classes, 1, 1, 1, 1, 0);
    bp_dropout_layer_init(classifier_ptr->dropout, 0.5);

    if(!LOAD_PRETRAINED_WEIGHT) bp_conv_layer_rand_weight(classifier_ptr->final_conv);
}

void BPSqueezeNet_init(BPSqueezeNet *squeeze_net_ptr, BPSqueezeNetVersion version,
        int num_classes) {
    squeeze_net_ptr->version     = version;
    squeeze_net_ptr->num_classes = num_classes;
    squeeze_net_ptr->features    = malloc(sizeof(BPFeaturesSequential));
    squeeze_net_ptr->classifier  = malloc(sizeof(BPClassifierSequential));
//    squeeze_net_ptr->output      = malloc(sizeof(Tensor));

    bp_features_sequential_init(squeeze_net_ptr->features, version);
    bp_classifier_sequential_init(squeeze_net_ptr->classifier, num_classes);
}

void bp_fire_forward(Tensor *input, BPFire *fire) {
    bp_conv_layer_forward(input, fire->squeeze, SAVE);
    //relu_forward(&(fire->squeeze->out));

    bp_conv_layer_forward(&(fire->squeeze->out), fire->expand1x1, SAVE);
    //relu_forward(&(fire->expand1x1->out));
    bp_conv_layer_forward(&(fire->squeeze->out), fire->expand3x3, SAVE);
    //relu_forward(&(fire->expand3x3->out));

    if (fire->output->data != NULL)
        free(fire->output->data);
    bp_tensor_cat(&(fire->expand1x1->out), &(fire->expand3x3->out), fire->output, 3);
}

void bp_features_forward(Tensor *input, BPFeaturesSequential *features) {
    bp_conv_layer_forward(input, features->conv, SAVE);
    //relu_forward(&(features->conv->out));
    bp_pool_layer_forward(&(features->conv->out), features->maxpool_list[0]);
    bp_fire_forward(&(features->maxpool_list[0]->out), features->fire_list[0]);
    bp_fire_forward(features->fire_list[0]->output, features->fire_list[1]);

    if(features->version == Version1_0){
        bp_fire_forward(features->fire_list[1]->output, features->fire_list[2]);
        bp_pool_layer_forward(features->fire_list[2]->output, features->maxpool_list[1]);
        bp_fire_forward(&(features->maxpool_list[1]->out), features->fire_list[3]);
        bp_fire_forward(features->fire_list[3]->output, features->fire_list[4]);
        bp_fire_forward(features->fire_list[4]->output, features->fire_list[5]);
        bp_fire_forward(features->fire_list[5]->output, features->fire_list[6]);
        bp_pool_layer_forward(features->fire_list[6]->output, features->maxpool_list[2]);
        bp_fire_forward(&(features->maxpool_list[2]->out), features->fire_list[7]);
    }else if(features->version == Version1_1){
        bp_pool_layer_forward(features->fire_list[1]->output, features->maxpool_list[1]);
        bp_fire_forward(&(features->maxpool_list[1]->out), features->fire_list[2]);
        bp_fire_forward(features->fire_list[2]->output, features->fire_list[3]);
        bp_pool_layer_forward(features->fire_list[3]->output, features->maxpool_list[2]);
        bp_fire_forward(&(features->maxpool_list[2]->out), features->fire_list[4]);
        bp_fire_forward(features->fire_list[4]->output, features->fire_list[5]);
        bp_fire_forward(features->fire_list[5]->output, features->fire_list[6]);
        bp_fire_forward(features->fire_list[6]->output, features->fire_list[7]);
    }else{
        fprintf(stderr, "\nWrong version of Squeeze Net\n");
    }

    if (features->output.data != NULL)
        free(features->output.data);
    features->output = bp_tensor_copy(features->fire_list[7]->output);
}

void bp_classification_forward(Tensor *input, BPClassifierSequential *classifier) {
    bp_dropout_layer_forward(input, classifier->dropout);
    bp_conv_layer_forward(input, classifier->final_conv, SAVE);
    //relu_forward(&(classifier->final_conv->out));
    bp_pool_layer_forward(&(classifier->final_conv->out), classifier->avgpool);

    if (classifier->output.data != NULL)
        free(classifier->output.data);
    classifier->output = bp_tensor_copy(&(classifier->avgpool->out));
}

void BPSqueezeNet_forward(Tensor *input, BPSqueezeNet *squeeze_net) {
    bp_features_forward(input, squeeze_net->features);
    bp_classification_forward(&(squeeze_net->features->output), squeeze_net->classifier);
    squeeze_net->output = squeeze_net->classifier->output;
}

void bp_fire_module_free(BPFire *fire_ptr){
    bp_conv_layer_free(fire_ptr->squeeze);
    bp_conv_layer_free(fire_ptr->expand1x1);
    bp_conv_layer_free(fire_ptr->expand3x3);
    bp_tensor_free(fire_ptr->output);

    free(fire_ptr->squeeze);
    free(fire_ptr->expand1x1);
    free(fire_ptr->expand3x3);
    free(fire_ptr->output);
}

void bp_features_free(BPFeaturesSequential *features_ptr){
    bp_conv_layer_free(features_ptr->conv);
    free(features_ptr->conv);
    for (int i = 0; i < 8; i++){
        bp_fire_module_free(features_ptr->fire_list[i]);
        free(features_ptr->fire_list[i]);
    }

    for (int i = 0; i < 3; i++){
        bp_pool_layer_free(features_ptr->maxpool_list[i]);
        free(features_ptr->maxpool_list[i]);
    }
    bp_tensor_free(&(features_ptr->output));
}

void bp_classifier_free(BPClassifierSequential *classifier_ptr){
    free(classifier_ptr->dropout);
    bp_conv_layer_free(classifier_ptr->final_conv);
    free(classifier_ptr->final_conv);
    bp_pool_layer_free(classifier_ptr->avgpool);
    free(classifier_ptr->avgpool);
    bp_tensor_free(&(classifier_ptr->output));
}

void BPSqueezeNet_free(BPSqueezeNet *squeeze_net) {
    bp_features_free(squeeze_net->features);
    free(squeeze_net->features);
    bp_classifier_free(squeeze_net->classifier);
    free(squeeze_net->classifier);
}
