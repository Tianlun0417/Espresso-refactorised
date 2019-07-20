#include "FloatTypeEspresso/FLOAT_ESP.h"
#include "FloatTypeEspresso/Cifar10Loader.h"


const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";

int main() {
    FloatTensor cifar_image = tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);

    inputLayer input_layer;
    cifar10_load(image_path, 0, 1, &cifar_image, &cifar_label);

    inputLayer_load(&cifar_image, &input_layer);
    inputLayer_forward(&input_layer);

    convLayer *conv = new_conv_layer(3, 64, 11, 11, 4, 4, 2);
    init_conv_layer(conv);
    convLayer_forward(&(input_layer.out), conv, 1);
    print_tensor(&(conv->out));
}
