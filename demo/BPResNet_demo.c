#include "CNN/BPResNet.h"


int main() {
    int num_classes = 10;
    __uint8_t *image = malloc(CIFAR_IMAGE_SIZE * sizeof(__uint8_t));
    __uint8_t *label = malloc(sizeof(__uint8_t));

//    int blocks[4] = {2, 2, 2, 2}; // ResNet-18
//    int blocks[4] = {3, 4, 6, 3}; // ResNet-50
//    int blocks[4] = {3, 4, 23, 3}; // ResNet-101
    int blocks[4] = {3, 8, 36, 3}; // ResNet-152

    BPResNet *resnet = malloc(sizeof(BPResNet));
    BPResNet_init(resnet, UseBasicBlock, blocks, num_classes);

    for (int idx = 0; idx < TEST_IMG; idx++) {
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


