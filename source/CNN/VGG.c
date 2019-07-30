#include "CNN/VGG.h"


const int VGGConfigA[] = {64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M};
const int VGGConfigB[] = {64, 64, M, 128, 128, M, 256, 256, M, 512, 512, M, 512, 512, M};
const int VGGConfigD[] = {64, 64, M, 128, 128, M, 256, 256, 256, M, 512, 512, 512, M, 512, 512, 512, M};
const int VGGConfigE[] = {64, 64, M, 128, 128, M, 256, 256, 256, 256, M, 512, 512, 512, 512, M, 512, 512, 512, 512, M};

void layers_config_init(VGGFeatures *features, size_t len, const int *config){
    int in_channels = 3;
    int maxpool_idx = 0;
    features->conv_count = 0;

    features->conv_list = malloc((len - 5) * sizeof(ConvLayer*));
    if (features->batch_norm)
        features->bnorm_list = malloc((len - 5) * sizeof(BnormLayer*));
    for (int i = 0; i < len; i++) {
        int v = config[i];
        if (v == M) {
            features->maxpool_list[maxpool_idx] = malloc(sizeof(PoolLayer));
            pool_layer_init(features->maxpool_list[maxpool_idx], 2, 2, 2, 2, 0, MAXPOOL);
            maxpool_idx++;
        } else {
            features->conv_list[features->conv_count] = malloc(sizeof(ConvLayer));
            conv_layer_init(features->conv_list[features->conv_count], in_channels, v, 3, 3, 1, 1, 1);

            if(!LOAD_PRETRAINED_WEIGHT)
                conv_layer_rand_weight(features->conv_list[features->conv_count]);

            if (features->batch_norm) {
                features->bnorm_list[features->conv_count] = malloc(sizeof(BnormLayer));
                bnorm_layer_init(features->bnorm_list[features->conv_count], v);

                if(!LOAD_PRETRAINED_WEIGHT)
                    bnorm_layer_rand_weight(features->bnorm_list[features->conv_count]);
            }
            if (features->conv_count >= len - 5)
                fprintf(stderr, "Conv layer init index outbound!\n");
            features->conv_count++;
        }
        if (v != M)
            in_channels = v;
    }
}

void features_init(VGGFeatures *features, bool batch_norm, VGGConfig config) {
    features->batch_norm = batch_norm;
    features->config = config;

    if (features->config == ConfigA)
        layers_config_init(features, sizeof(VGGConfigA) / sizeof(VGGConfigA[0]), VGGConfigA);
    else if (features->config == ConfigB)
        layers_config_init(features, sizeof(VGGConfigB) / sizeof(VGGConfigB[0]), VGGConfigB);
    else if (features->config == ConfigD)
        layers_config_init(features, sizeof(VGGConfigD) / sizeof(VGGConfigD[0]), VGGConfigD);
    else if (features->config == ConfigE)
        layers_config_init(features, sizeof(VGGConfigE) / sizeof(VGGConfigE[0]), VGGConfigE);
    else
        fprintf(stderr, "Unkonwn config!");

}

void classifier_init(VGGClassifier *classifier, int num_classes) {
    classifier->dense1   = malloc(sizeof(DenseLayer));
    classifier->dense2   = malloc(sizeof(DenseLayer));
    classifier->dense3   = malloc(sizeof(DenseLayer));
    classifier->dropout1 = malloc(sizeof(DropoutLayer));
    classifier->dropout2 = malloc(sizeof(DropoutLayer));
    dense_layer_init(classifier->dense1, 4096, 512);
    dense_layer_init(classifier->dense2, 4096, 4096);
    dense_layer_init(classifier->dense3, num_classes, 4096);
    dropout_layer_init(classifier->dropout1, 0.5);
    dropout_layer_init(classifier->dropout2, 0.5);

    if(!LOAD_PRETRAINED_WEIGHT){
        dense_layer_rand_weight(classifier->dense1);
        dense_layer_rand_weight(classifier->dense2);
        dense_layer_rand_weight(classifier->dense3);
    }
}

void VGG_init(VGG *vgg, VGGConfig config, int num_classes, bool batch_norm) {
    vgg->config     = config;
    vgg->batch_norm = batch_norm;
    vgg->features   = malloc(sizeof(VGGFeatures));
    vgg->classifier = malloc(sizeof(VGGClassifier));

    features_init(vgg->features, batch_norm, config);
    classifier_init(vgg->classifier, num_classes);
}

void layers_config_forward(Tensor *input, VGGFeatures *features, size_t len, const int *config){
    Tensor *tmp_result = input;
    int maxpool_idx = 0;
    int conv_idx = 0;

    for (int i = 0; i < len; i++) {
        int v = config[i];
        if (v == M){
            pool_layer_forward(tmp_result, features->maxpool_list[maxpool_idx]);
            tmp_result = &(features->maxpool_list[maxpool_idx]->out);
            maxpool_idx++;
        }else{
            conv_layer_forward(tmp_result, features->conv_list[conv_idx], SAVE);
            if (features->batch_norm){
                bnorm_layer_forward(&(features->conv_list[conv_idx]->out),
                                    features->bnorm_list[conv_idx], SAVE);
            }
            relu_forward(&(features->conv_list[conv_idx]->out));
            tmp_result = &(features->conv_list[conv_idx]->out);
            conv_idx++;
        }
    }

    if (features->output.data != NULL)
        free(features->output.data);
    features->output = tensor_copy(tmp_result);
}

void features_forward(Tensor *input, VGGFeatures *features) {
    if (features->config == ConfigA)
        layers_config_forward(input, features, sizeof(VGGConfigA) / sizeof(VGGConfigA[0]),
                VGGConfigA);
    else if (features->config == ConfigB)
        layers_config_forward(input, features, sizeof(VGGConfigB) / sizeof(VGGConfigB[0]),
                VGGConfigB);
    else if (features->config == ConfigD)
        layers_config_forward(input, features, sizeof(VGGConfigD) / sizeof(VGGConfigD[0]),
                VGGConfigD);
    else if (features->config == ConfigE)
        layers_config_forward(input, features, sizeof(VGGConfigE) / sizeof(VGGConfigE[0]),
                VGGConfigE);
    else
        fprintf(stderr, "Unkonwn config!");
}

void classifier_forward(Tensor *input, VGGClassifier *classifier) {
    dense_layer_forward(input, classifier->dense1, SAVE);
    dropout_layer_forward(&(classifier->dense1->out), classifier->dropout1);
    dense_layer_forward(&(classifier->dense1->out), classifier->dense2, SAVE);
    dropout_layer_forward(&(classifier->dense2->out), classifier->dropout2);
    dense_layer_forward(&(classifier->dense2->out), classifier->dense3, SAVE);

    if (classifier->output.data != NULL)
        free(classifier->output.data);
    classifier->output = tensor_copy(&(classifier->dense3->out));
}

void VGG_forward(Tensor *input, VGG *vgg) {
    features_forward(input, vgg->features);
    classifier_forward(&(vgg->features->output), vgg->classifier);
    if (vgg->output.data != NULL)
        free(vgg->output.data);
    vgg->output = tensor_copy(&(vgg->classifier->output));
}

void features_free(VGGFeatures* features){
    for (int i = 0; i < features->conv_count; i++){
        conv_layer_free(features->conv_list[i]);
        if (features->batch_norm){
            bnorm_layer_free(features->bnorm_list[i]);
            free(features->bnorm_list[i]);
        }
        free(features->conv_list[i]);
    }
    for (int i = 0; i < 5; i++){
        pool_layer_free(features->maxpool_list[i]);
        free(features->maxpool_list[i]);
    }

    free(features->conv_list);
    free(features->bnorm_list);
    tensor_free(&(features->output));
}

void classifier_free(VGGClassifier *classifier){
    dense_layer_free(classifier->dense1);
    dense_layer_free(classifier->dense2);
    dense_layer_free(classifier->dense3);

    free(classifier->dense1);
    free(classifier->dense2);
    free(classifier->dense3);

    free(classifier->dropout1);
    free(classifier->dropout2);

    tensor_free(&(classifier->output));
}

void VGG_free(VGG *vgg) {
    features_free(vgg->features);
    classifier_free(vgg->classifier);

    free(vgg->features);
    free(vgg->classifier);

    tensor_free(&(vgg->output));
}


