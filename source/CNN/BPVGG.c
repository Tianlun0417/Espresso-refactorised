#include "CNN/BPVGG.h"


const int VGGConfigA[] = {64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M};
const int VGGConfigB[] = {64, 64, M, 128, 128, M, 256, 256, M, 512, 512, M, 512, 512, M};
const int VGGConfigD[] = {64, 64, M, 128, 128, M, 256, 256, 256, M, 512, 512, 512, M, 512, 512, 512, M};
const int VGGConfigE[] = {64, 64, M, 128, 128, M, 256, 256, 256, 256, M, 512, 512, 512, 512, M, 512, 512, 512, 512, M};

void bp_layers_config_init(BPVGGFeatures *features, size_t len, const int *config){
    int in_channels = 3;
    int maxpool_idx = 0;
    features->conv_count = 0;

    features->conv_list = malloc((len - 5) * sizeof(BPConvLayer*));
    if (features->batch_norm)
        features->bnorm_list = malloc((len - 5) * sizeof(BPBnormLayer*));
    for (int i = 0; i < len; i++) {
        int v = config[i];
        if (v == M) {
            features->maxpool_list[maxpool_idx] = malloc(sizeof(BPPoolLayer));
            bp_pool_layer_init(features->maxpool_list[maxpool_idx], 2, 2, 2, 2, 0, MAXPOOL);
            maxpool_idx++;
        } else {
            features->conv_list[features->conv_count] = malloc(sizeof(BPConvLayer));
            bp_conv_layer_init(features->conv_list[features->conv_count], in_channels, v, 3, 3, 1, 1, 1);

            if(!LOAD_PRETRAINED_WEIGHT)
                bp_conv_layer_rand_weight(features->conv_list[features->conv_count]);

            if (features->batch_norm) {
                features->bnorm_list[features->conv_count] = malloc(sizeof(BPBnormLayer));
                bp_bnorm_layer_init(features->bnorm_list[features->conv_count], v);

                if(!LOAD_PRETRAINED_WEIGHT)
                    bp_bnorm_layer_rand_weight(features->bnorm_list[features->conv_count]);
            }
            if (features->conv_count >= len - 5)
                fprintf(stderr, "Conv layer init index outbound!\n");
            features->conv_count++;
        }
        if (v != M)
            in_channels = v;
    }
}

void bp_features_init(BPVGGFeatures *features, bool batch_norm, BPVGGConfig config) {
    features->batch_norm = batch_norm;
    features->config = config;

    if (features->config == ConfigA)
        bp_layers_config_init(features, sizeof(VGGConfigA) / sizeof(VGGConfigA[0]), VGGConfigA);
    else if (features->config == ConfigB)
        bp_layers_config_init(features, sizeof(VGGConfigB) / sizeof(VGGConfigB[0]), VGGConfigB);
    else if (features->config == ConfigD)
        bp_layers_config_init(features, sizeof(VGGConfigD) / sizeof(VGGConfigD[0]), VGGConfigD);
    else if (features->config == ConfigE)
        bp_layers_config_init(features, sizeof(VGGConfigE) / sizeof(VGGConfigE[0]), VGGConfigE);
    else
        fprintf(stderr, "Unkonwn config!");
}

void bp_classifier_init(BPVGGClassifier *classifier, int num_classes) {
    classifier->dense1   = malloc(sizeof(BPDenseLayer));
    classifier->dense2   = malloc(sizeof(BPDenseLayer));
    classifier->dense3   = malloc(sizeof(BPDenseLayer));
    classifier->dropout1 = malloc(sizeof(BPDropoutLayer));
    classifier->dropout2 = malloc(sizeof(BPDropoutLayer));
    bp_dense_layer_init(classifier->dense1, 4096, 512);
    bp_dense_layer_init(classifier->dense2, 4096, 4096);
    bp_dense_output_layer_init(classifier->dense3, num_classes, 4096);
    bp_dropout_layer_init(classifier->dropout1, 0.5);
    bp_dropout_layer_init(classifier->dropout2, 0.5);

    if(!LOAD_PRETRAINED_WEIGHT){
        bp_dense_layer_rand_weight(classifier->dense1);
        bp_dense_layer_rand_weight(classifier->dense2);
        bp_dense_output_layer_rand_weight(classifier->dense3);
    }
}

void BPVGG_init(BPVGG *vgg, BPVGGConfig config, int num_classes, bool batch_norm) {
    vgg->config     = config;
    vgg->batch_norm = batch_norm;
    vgg->features   = malloc(sizeof(BPVGGFeatures));
    vgg->classifier = malloc(sizeof(BPVGGClassifier));

    bp_features_init(vgg->features, batch_norm, config);
    bp_classifier_init(vgg->classifier, num_classes);
}

void bp_layers_config_forward(Tensor *input, BPVGGFeatures *features, size_t len, const int *config){
    Tensor *tmp_result = input;
    int maxpool_idx = 0;
    int conv_idx = 0;

    for (int i = 0; i < len; i++) {
        int v = config[i];
        if (v == M){
            bp_pool_layer_forward(tmp_result, features->maxpool_list[maxpool_idx]);
            tmp_result = &(features->maxpool_list[maxpool_idx]->out);
            maxpool_idx++;
        }else{
            bp_conv_layer_forward(tmp_result, features->conv_list[conv_idx], SAVE);
            if (features->batch_norm){
                bp_bnorm_layer_forward(&(features->conv_list[conv_idx]->out),
                                    features->bnorm_list[conv_idx], SAVE);
            }
            //relu_forward(&(features->conv_list[conv_idx]->out));
            tmp_result = &(features->conv_list[conv_idx]->out);
            conv_idx++;
        }
    }

    if (features->output.data != NULL)
        free(features->output.data);
    features->output = bp_tensor_copy(tmp_result);
}

void bp_features_forward(Tensor *input, BPVGGFeatures *features) {
    if (features->config == ConfigA)
        bp_layers_config_forward(input, features, sizeof(VGGConfigA) / sizeof(VGGConfigA[0]),
                              VGGConfigA);
    else if (features->config == ConfigB)
        bp_layers_config_forward(input, features, sizeof(VGGConfigB) / sizeof(VGGConfigB[0]),
                              VGGConfigB);
    else if (features->config == ConfigD)
        bp_layers_config_forward(input, features, sizeof(VGGConfigD) / sizeof(VGGConfigD[0]),
                              VGGConfigD);
    else if (features->config == ConfigE)
        bp_layers_config_forward(input, features, sizeof(VGGConfigE) / sizeof(VGGConfigE[0]),
                              VGGConfigE);
    else
        fprintf(stderr, "Unkonwn config!");
}

void bp_classifier_forward(Tensor *input, BPVGGClassifier *classifier) {
    bp_dense_layer_forward(input, classifier->dense1, SAVE);
    bp_dropout_layer_forward(&(classifier->dense1->out), classifier->dropout1);
    bp_dense_layer_forward(&(classifier->dense1->out), classifier->dense2, SAVE);
    bp_dropout_layer_forward(&(classifier->dense2->out), classifier->dropout2);
    bp_dense_output_layer_forward(&(classifier->dense2->out), classifier->dense3, SAVE);

//    if (classifier->output)
//        free(classifier->output);
    classifier->output = classifier->dense3->output_arr;
}

void BPVGG_forward(Tensor *input, BPVGG *vgg) {
    bp_features_forward(input, vgg->features);
    bp_classifier_forward(&(vgg->features->output), vgg->classifier);
//    if (vgg->output)
//        free(vgg->output);
    vgg->output = vgg->classifier->output;
}

void bp_features_free(BPVGGFeatures* features){
    for (int i = 0; i < features->conv_count; i++){
        bp_conv_layer_free(features->conv_list[i]);
        if (features->batch_norm){
            bp_bnorm_layer_free(features->bnorm_list[i]);
            free(features->bnorm_list[i]);
        }
        free(features->conv_list[i]);
    }
    for (int i = 0; i < 5; i++){
        bp_pool_layer_free(features->maxpool_list[i]);
        free(features->maxpool_list[i]);
    }

    free(features->conv_list);
    free(features->bnorm_list);
    bp_tensor_free(&(features->output));
}

void bp_classifier_free(BPVGGClassifier *classifier){
    bp_dense_layer_free(classifier->dense1);
    bp_dense_layer_free(classifier->dense2);
    bp_dense_output_layer_free(classifier->dense3);

    free(classifier->dense1);
    free(classifier->dense2);
    free(classifier->dense3);

    free(classifier->dropout1);
    free(classifier->dropout2);

    //free(classifier->output);
}


void BPVGG_free(BPVGG *vgg) {
    bp_features_free(vgg->features);
    bp_classifier_free(vgg->classifier);

    free(vgg->features);
    free(vgg->classifier);
    free(vgg->output);
}
