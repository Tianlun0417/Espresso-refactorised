#include "CNN/AlexNet.h"

void new_features(Features *features) {
    features->conv1 = malloc(sizeof(ConvLayer));
    features->conv2 = malloc(sizeof(ConvLayer));
    features->conv3 = malloc(sizeof(ConvLayer));
    features->conv4 = malloc(sizeof(ConvLayer));
    features->conv5 = malloc(sizeof(ConvLayer));

    conv_layer_init(features->conv1, 3,   64,  11, 11, 4, 4, 2);
    conv_layer_init(features->conv2, 64,  192, 5,  5,  1, 1, 2);
    conv_layer_init(features->conv3, 192, 384, 3,  3,  1, 1, 1);
    conv_layer_init(features->conv4, 384, 256, 3,  3,  1, 1, 1);
    conv_layer_init(features->conv5, 256, 256, 3,  3,  1, 1, 1);

    if(!LOAD_PRETRAINED_WEIGHT){
        conv_layer_rand_weight(features->conv1);
        conv_layer_rand_weight(features->conv2);
        conv_layer_rand_weight(features->conv3);
        conv_layer_rand_weight(features->conv4);
        conv_layer_rand_weight(features->conv5);
    }

    features->maxpool1 = malloc(sizeof(PoolLayer));
    features->maxpool2 = malloc(sizeof(PoolLayer));
    features->maxpool3 = malloc(sizeof(PoolLayer));
    new_pool_layer(features->maxpool1, 3, 3, 2, 2, 0, MAXPOOL);
    new_pool_layer(features->maxpool2, 3, 3, 2, 2, 0, MAXPOOL);
    new_pool_layer(features->maxpool3, 3, 3, 2, 2, 0, MAXPOOL);
}

void new_classifier(Classifier *classifier, int num_classes) {
    classifier->num_classes = num_classes;
    classifier->dropout = new_dropout_layer(0.5);
    classifier->dense1  = new_dense_layer(4096, 256);
    classifier->dense2  = new_dense_layer(4096, 4096);
    classifier->dense3  = new_dense_layer(classifier->num_classes, 4096);

    if(!LOAD_PRETRAINED_WEIGHT){
        dense_layer_rand_weight(classifier->dense1);
        dense_layer_rand_weight(classifier->dense2);
        dense_layer_rand_weight(classifier->dense3);
    }
}

void AlexNet_init(AlexNet *alex_net, int num_classes) {
    alex_net->num_classes = num_classes;
    alex_net->features = malloc(sizeof(Features));
    alex_net->classifier = malloc(sizeof(Classifier));

    new_features(alex_net->features);
    new_classifier(alex_net->classifier, alex_net->num_classes);
}

void features_forward(Tensor *input, Features *features) {
    conv_layer_forward(input, features->conv1, SAVE);
    relu_forward(&(features->conv1->out));
    pool_layer_forward(&(features->conv1->out), features->maxpool1);
    conv_layer_forward(&(features->maxpool1->out), features->conv2, SAVE);
    relu_forward(&(features->conv2->out));
    pool_layer_forward(&(features->conv2->out), features->maxpool2);

    conv_layer_forward(&(features->maxpool2->out), features->conv3, SAVE);
    relu_forward(&(features->conv3->out));
    conv_layer_forward(&(features->conv3->out), features->conv4, SAVE);
    relu_forward(&(features->conv4->out));
    conv_layer_forward(&(features->conv4->out), features->conv5, SAVE);
    relu_forward(&(features->conv5->out));

    if (features->output.data != NULL)
        free(features->output.data);
    features->output = tensor_copy(&(features->conv5->out));
}

void classifier_forward(Tensor *input, Classifier *classifier) {
    dropout_layer_forward(input, classifier->dropout);
    dense_layer_forward(input, classifier->dense1, SAVE);
    relu_forward(&(classifier->dense1->out));
    dropout_layer_forward(&(classifier->dense1->out), classifier->dropout);
    dense_layer_forward(&(classifier->dense1->out), classifier->dense2, SAVE);
    relu_forward(&(classifier->dense2->out));
    dense_layer_forward(&(classifier->dense2->out), classifier->dense3, SAVE);

    if (classifier->output.data != NULL)
        free(classifier->output.data);
    classifier->output = tensor_copy(&(classifier->dense3->out));
}

void alexnet_forward(Tensor *input, AlexNet *alex_net) {
    features_forward(input, alex_net->features);
    classifier_forward(&(alex_net->features->output), alex_net->classifier);
    if (alex_net->output.data != NULL)
        free(alex_net->output.data);
    alex_net->output = tensor_copy(&(alex_net->classifier->output));
}

void features_free(Features *features){
    conv_layer_free(features->conv1);
    conv_layer_free(features->conv2);
    conv_layer_free(features->conv3);
    conv_layer_free(features->conv4);
    conv_layer_free(features->conv5);
    poolLayer_free(features->maxpool1);
    poolLayer_free(features->maxpool2);
    poolLayer_free(features->maxpool3);
    tensor_free(&(features->output));

    free(features->conv1);
    free(features->conv2);
    free(features->conv3);
    free(features->conv4);
    free(features->conv5);
    free(features->maxpool1);
    free(features->maxpool2);
    free(features->maxpool3);
}

void classifier_free(Classifier *classifier){
    denseLayer_free(classifier->dense1);
    denseLayer_free(classifier->dense2);
    denseLayer_free(classifier->dense3);
    tensor_free(&(classifier->output));

    free(classifier->dense1);
    free(classifier->dense2);
    free(classifier->dense3);
    free(classifier->dropout);
}

void AlexNet_free(AlexNet *alexnet) {
    features_free(alexnet->features);
    classifier_free(alexnet->classifier);
    tensor_free(&(alexnet->output));

    free(alexnet->features);
    free(alexnet->classifier);
}


