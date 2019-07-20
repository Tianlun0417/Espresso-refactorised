#include "CNN/AlexNet.h"

Features *new_features() {
    Features *features = (Features*) malloc(sizeof(Features));

    features->conv1 = new_conv_layer(3,   64,  11, 11, 4, 4, 2);
    features->conv2 = new_conv_layer(64,  192, 5,  5,  1, 1, 2);
    features->conv3 = new_conv_layer(192, 384, 3,  3,  1, 1, 1);
    features->conv4 = new_conv_layer(384, 256, 3,  3,  1, 1, 1);
    features->conv5 = new_conv_layer(256, 256, 3,  3,  1, 1, 1);

    if(!LOAD_PRETRAINED_WEIGHT){
        init_conv_layer(features->conv1);
        init_conv_layer(features->conv2);
        init_conv_layer(features->conv3);
        init_conv_layer(features->conv4);
        init_conv_layer(features->conv5);
    }

    features->maxpool1 = new_pool_layer(3, 3, 2, 2, 0, MAXPOOL);
    features->maxpool2 = new_pool_layer(3, 3, 2, 2, 0, MAXPOOL);
    features->maxpool3 = new_pool_layer(3, 3, 2, 2, 0, MAXPOOL);
    features->output = NULL;

    return features;
}

Classifier *new_classifier(int num_classes) {
    Classifier *classifier = (Classifier*) malloc(sizeof(Classifier));

    classifier->num_classes = num_classes;
    classifier->dropout = new_dropout_layer(0.5);
    classifier->dense1  = new_dense_layer(256*6*6, 4096);
    classifier->dense2  = new_dense_layer(4096, 4096);
    classifier->dense3  = new_dense_layer(4096, classifier->num_classes);

    if(!LOAD_PRETRAINED_WEIGHT){
        init_dense_layer(classifier->dense1);
        init_dense_layer(classifier->dense2);
        init_dense_layer(classifier->dense3);
    }

    classifier->output  = NULL;

    return classifier;
}

AlexNet *AlexNet_init(int num_classes) {
    AlexNet *alex_net = (AlexNet*) malloc(sizeof(AlexNet));

    alex_net->num_classes = num_classes;
    alex_net->features = new_features();
    alex_net->classifier = new_classifier(alex_net->num_classes);
    alex_net->output = NULL;

    return alex_net;
}

void features_forward(Tensor *input, Features *features) {
    convLayer_forward(input, features->conv1, SAVE);
    reluAct_forward(&(features->conv1->out));
    poolLayer_forward(&(features->conv1->out), features->maxpool1);
    convLayer_forward(&(features->maxpool1->out), features->conv2, SAVE);
    reluAct_forward(&(features->conv2->out));
    poolLayer_forward(&(features->conv2->out), features->maxpool2);

    convLayer_forward(&(features->maxpool2->out), features->conv3, SAVE);
    reluAct_forward(&(features->conv3->out));
    convLayer_forward(&(features->conv3->out), features->conv4, SAVE);
    reluAct_forward(&(features->conv4->out));
    convLayer_forward(&(features->conv4->out), features->conv5, SAVE);
    reluAct_forward(&(features->conv5->out));
    poolLayer_forward(&(features->conv5->out), features->maxpool3);

    features->output = &(features->maxpool3->out);
}

void classifier_forward(Tensor *input, Classifier *classifier) {
    dropoutLayer_forward(input, classifier->dropout);
    denseLayer_forward(input, classifier->dense1, SAVE);
    reluAct_forward(&(classifier->dense1->out));
    dropoutLayer_forward(&(classifier->dense1->out), classifier->dropout);
    denseLayer_forward(&(classifier->dense1->out), classifier->dense2, SAVE);
    reluAct_forward(&(classifier->dense2->out));
    denseLayer_forward(&(classifier->dense2->out), classifier->dense3, SAVE);

    classifier->output = &(classifier->dense3->out);
}

void alexnet_forward(Tensor *input, AlexNet *alex_net) {
    features_forward(input, alex_net->features);
    classifier_forward(alex_net->features->output, alex_net->classifier);
    alex_net->output = alex_net->classifier->output;
}


