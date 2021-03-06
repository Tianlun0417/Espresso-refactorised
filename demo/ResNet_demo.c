#include "CNN/ResNet.h"
#include "DataLoader/Cifar10Loader.h"


int main() {
    FloatTensor cifar_image = tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);

    InputLayer input_layer;
//    int blocks[4] = {2, 2, 2, 2}; // ResNet-18
    //int blocks[4] = {3, 4, 6, 3}; // ResNet-50
    //int blocks[4] = {3, 4, 23, 3}; // ResNet-101
    int blocks[4] = {3, 8, 36, 3}; // ResNet-152

    ResNet *resnet = malloc(sizeof(ResNet));
    ResNet_init(resnet, UseBasicBlock, blocks, 10);

    for (int idx = 0; idx < TEST_IMG; idx++) {
        cifar10_load(image_path, idx, 1, &cifar_image, &cifar_label);

        input_layer_load(&cifar_image, &input_layer);
        input_layer_forward(&input_layer);

        ResNet_forward(&(input_layer.out), resnet);
        printf("NO.%d ", idx);
        print_tensor(&resnet->output);
    }

    tensor_free(&cifar_label);
    input_layer_free(&input_layer);
    ResNet_free(resnet);
    free(resnet);
}