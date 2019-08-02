#include <CNN/DenseNet.h>
#include "FloatTypeEspresso/FLOAT_ESP.h"
#include "DataLoader/Cifar10Loader.h"


const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";

int main(){
    int growth_rate = 32;
    int num_init_features = 64;

//    int block_config[4] = {6, 12, 24, 16}; // DenseNet-121
//    int block_config[4] = {6, 12, 32, 32}; // DenseNet-169
//    int block_config[4] = {6, 12, 48, 32}; // DenseNet-201
    int block_config[4] = {6, 12, 36, 24}; // DenseNet-161

    int bn_size = 4;
    float drop_rate = 0.5;
    int num_classes = 10;

    FloatTensor cifar_image = tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);

    InputLayer input_layer;

    DenseNet *densenet = malloc(sizeof(DenseNet));
    DenseNet_init(densenet, block_config, num_init_features, growth_rate, bn_size, drop_rate, num_classes);

    for (int idx = 0; idx < TEST_IMG; idx++) {
        cifar10_load(image_path, idx, 1, &cifar_image, &cifar_label);

        input_layer_load(&cifar_image, &input_layer);
        input_layer_forward(&input_layer);

        DenseNet_forward(&(input_layer.out), densenet);
        printf("NO.%d ", idx);
        print_tensor(&densenet->output);
    }

    tensor_free(&cifar_label);
    input_layer_free(&input_layer);
    DenseNet_free(densenet);
    free(densenet);

    return 0;
}