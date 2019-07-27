#include "CNN/BPAlexNet.h"

void new_features(Features *features) {
    features->conv1 = malloc(sizeof(BPConvLayer));
    features->conv2 = malloc(sizeof(BPConvLayer));
    features->conv3 = malloc(sizeof(BPConvLayer));
    features->conv4 = malloc(sizeof(BPConvLayer));
    features->conv5 = malloc(sizeof(BPConvLayer));

    bp_conv_layer_init(features->conv1, 3,   64,  11, 11, 4, 4, 2);
    bp_conv_layer_init(features->conv2, 64,  192, 5,  5,  1, 1, 2);
    bp_conv_layer_init(features->conv3, 192, 384, 3,  3,  1, 1, 1);
    bp_conv_layer_init(features->conv4, 384, 256, 3,  3,  1, 1, 1);
    bp_conv_layer_init(features->conv5, 256, 256, 3,  3,  1, 1, 1);

    if(!LOAD_PRETRAINED_WEIGHT){
        bp_conv_layer_rand_weight(features->conv1);
        bp_conv_layer_rand_weight(features->conv2);
        bp_conv_layer_rand_weight(features->conv3);
        bp_conv_layer_rand_weight(features->conv4);
        bp_conv_layer_rand_weight(features->conv5);
    }

    features->maxpool1 = malloc(sizeof(BPPoolLayer));
    features->maxpool2 = malloc(sizeof(BPPoolLayer));
    features->maxpool3 = malloc(sizeof(BPPoolLayer));
    bp_pool_layer_init(features->maxpool1, 3, 3, 2, 2, 0, MAXPOOL);
    bp_pool_layer_init(features->maxpool2, 3, 3, 2, 2, 0, MAXPOOL);
    bp_pool_layer_init(features->maxpool3, 3, 3, 2, 2, 0, MAXPOOL);
}

void new_classifier(Classifier *classifier, int num_classes) {
    classifier->num_classes = num_classes;
    classifier->dropout = malloc(sizeof(BPDropoutLayer));
    classifier->dense1  = malloc(sizeof(BPDenseLayer));
    classifier->dense2  = malloc(sizeof(BPDenseLayer));
    classifier->dense3  = malloc(sizeof(BPDenseOutputLayer));

    bp_dropout_layer_init(classifier->dropout, 0.5);
    bp_dense_layer_init(classifier->dense1, 4096, 256);
    bp_dense_layer_init(classifier->dense2, 4096, 4096);
    bp_dense_output_layer_init(classifier->dense3, classifier->num_classes, 4096);

    if(!LOAD_PRETRAINED_WEIGHT){
        bp_dense_layer_rand_weight(classifier->dense1);
        bp_dense_layer_rand_weight(classifier->dense2);
        bp_dense_output_layer_rand_weight(classifier->dense3);
    }
}

void BPAlexNet_init(BPAlexNet *alex_net, int num_classes) {
    alex_net->num_classes = num_classes;
    alex_net->features = malloc(sizeof(Features));
    alex_net->classifier = malloc(sizeof(Classifier));

    new_features(alex_net->features);
    new_classifier(alex_net->classifier, alex_net->num_classes);
}

void features_forward(Tensor *input, Features *features) {
    bp_conv_layer_forward(input, features->conv1, SAVE);
    //relu_forward(&(features->conv1->out));
    bp_pool_layer_forward(&(features->conv1->out), features->maxpool1);
    bp_conv_layer_forward(&(features->maxpool1->out), features->conv2, SAVE);
    //relu_forward(&(features->conv2->out));
    bp_pool_layer_forward(&(features->conv2->out), features->maxpool2);

    bp_conv_layer_forward(&(features->maxpool2->out), features->conv3, SAVE);
    //relu_forward(&(features->conv3->out));
    bp_conv_layer_forward(&(features->conv3->out), features->conv4, SAVE);
    //relu_forward(&(features->conv4->out));
    bp_conv_layer_forward(&(features->conv4->out), features->conv5, SAVE);
    //relu_forward(&(features->conv5->out));

    if (features->output.data != NULL)
        free(features->output.data);
    features->output = bp_tensor_copy(&(features->conv5->out));
}

void classifier_forward(Tensor *input, Classifier *classifier) {
    bp_dropout_layer_forward(input, classifier->dropout);
    bp_dense_layer_forward(input, classifier->dense1, SAVE);
    //relu_forward(&(classifier->dense1->out));
    bp_dropout_layer_forward(&(classifier->dense1->out), classifier->dropout);
    bp_dense_layer_forward(&(classifier->dense1->out), classifier->dense2, SAVE);
    //relu_forward(&(classifier->dense2->out));
    bp_dense_output_layer_forward(&(classifier->dense2->out), classifier->dense3, SAVE);

    classifier->output = classifier->dense3->output_arr;
}

void BPAlexNet_forward(Tensor *input, BPAlexNet *alex_net) {
    features_forward(input, alex_net->features);
    classifier_forward(&(alex_net->features->output), alex_net->classifier);
    alex_net->output = alex_net->classifier->output;
}

void features_free(Features *features){
    bp_conv_layer_free(features->conv1);
    bp_conv_layer_free(features->conv2);
    bp_conv_layer_free(features->conv3);
    bp_conv_layer_free(features->conv4);
    bp_conv_layer_free(features->conv5);
    bp_poolLayer_free(features->maxpool1);
    bp_poolLayer_free(features->maxpool2);
    bp_poolLayer_free(features->maxpool3);
    bp_tensor_free(&(features->output));

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
    bp_denseLayer_free(classifier->dense1);
    bp_denseLayer_free(classifier->dense2);
    bp_dense_output_layer_free(classifier->dense3);

    free(classifier->dense1);
    free(classifier->dense2);
    free(classifier->dense3);
    free(classifier->dropout);
}

void BPAlexNet_free(BPAlexNet *alexnet) {
    features_free(alexnet->features);
    classifier_free(alexnet->classifier);

    free(alexnet->features);
    free(alexnet->classifier);
    free(alexnet->output);
}