#include "CNN/BPResNet.h"


const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";

int main() {
    int num_classes = 10;
    __uint8_t *image = malloc(CIFAR_IMAGE_SIZE * sizeof(__uint8_t));
    __uint8_t *label = malloc(sizeof(__uint8_t));

    int blocks[4] = {2, 2, 2, 2};

    BPResNet *resnet = malloc(sizeof(BPResNet));
    BPResNet_init(resnet, UseBasicBlock, blocks, num_classes);

    for (int idx = 0; idx < 10000; idx++) {
        cifar10_load_int(image_path, idx, 1, image, label);
        BPTensor image_tensor = bp_tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
        bp_input_layer_forward(image, &image_tensor);

        BPResNet_forward(&image_tensor, resnet);

        printf("NO.%d ", idx);
        for (int i = 0; i < num_classes; i++)
            printf(" %f", resnet->output[i]);
        puts("");

        bp_tensor_free(&image_tensor);
    }

    free(image);
    free(label);
    BPResNet_free(resnet);
    free(resnet);
}


