#include "CNN/BPDenseNet.h"


const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";

int main() {
    int growth_rate = 32;

//    int block_config[4] = {6, 12, 24, 16}; // DenseNet-121
//    int block_config[4] = {6, 12, 32, 32}; // DenseNet-169
    int block_config[4] = {6, 12, 48, 32}; // DenseNet-201
//    int block_config[4] = {6, 12, 36, 24}; // DenseNet-161

    int num_init_features = 64;
    int bn_size = 4;
    float drop_rate = 0.5;
    int num_classes = 10;

    __uint8_t *image = malloc(CIFAR_IMAGE_SIZE * sizeof(__uint8_t));
    __uint8_t *label = malloc(sizeof(__uint8_t));

    BPDenseNet *dense_net = malloc(sizeof(BPDenseNet));
    BPDenseNet_init(dense_net, block_config, num_init_features, growth_rate, bn_size, drop_rate, num_classes);

    for (int idx = 0; idx < TEST_IMG; idx++) {
        cifar10_load_int(image_path, idx, 1, image, label);
        BPTensor image_tensor = bp_tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
        bp_input_layer_forward(image, &image_tensor);

        BPDenseNet_forward(&image_tensor, dense_net);

        printf("NO.%d ", idx);
        for (int i = 0; i < num_classes; i++)
            printf(" %f", dense_net->output[i]);
        puts("");

        bp_tensor_free(&image_tensor);
    }

    free(image);
    free(label);
    BPDenseNet_free(dense_net);
    free(dense_net);
}


