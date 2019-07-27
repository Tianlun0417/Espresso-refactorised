#include <zconf.h>
#include "FloatTypeEspresso/FLOAT_ESP.h"
#include "DataLoader/Cifar10Loader.h"


const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";

int main() {
    FloatTensor cifar_image = tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);

    InputLayer input_layer;
    cifar10_load(image_path, 0, 1, &cifar_image, &cifar_label);

    input_layer_load(&cifar_image, &input_layer);
    input_layer_forward(&input_layer);

    ConvLayer *conv1 = new_conv_layer(3,   64,  11, 11, 4, 4, 2);
    ConvLayer *conv2 = new_conv_layer(64,  192, 5,  5,  1, 1, 2);
    ConvLayer *conv3 = new_conv_layer(192, 384, 3,  3,  1, 1, 1);
    ConvLayer *conv4 = new_conv_layer(384, 256, 3,  3,  1, 1, 1);
    ConvLayer *conv5 = new_conv_layer(256, 256, 3,  3,  1, 1, 1);
    PoolLayer *pool1 = new_pool_layer(3, 3, 2, 2, 0, MAXPOOL);
    PoolLayer *pool2 = new_pool_layer(3, 3, 2, 2, 0, MAXPOOL);
    DenseLayer *dense = new_dense_layer(10, 49 * 256);
    init_conv_layer(conv1);
    init_conv_layer(conv2);
    init_conv_layer(conv3);
    init_conv_layer(conv4);
    init_conv_layer(conv5);
    init_dense_layer(dense);

    //print_tensor(&(input_layer.out));
    conv_layer_forward(&(input_layer.out), conv1, 1);
    relu_forward(&(conv1->out));
    pool_layer_forward(&(conv1->out), pool1);
    conv_layer_forward(&(pool1->out), conv2, 1);
    relu_forward(&(conv2->out));

    pool_layer_forward(&(conv2->out), pool1);
//    pool_layer_forward(&(pool1->out), pool2);

    conv_layer_forward(&(pool1->out), conv3, 1);
    relu_forward(&(conv3->out));
    conv_layer_forward(&(conv3->out), conv4, 1);
    relu_forward(&(conv4->out));
    conv_layer_forward(&(conv4->out), conv5, 1);
    relu_forward(&(conv5->out));
    pool_layer_forward(&(conv5->out), pool1);
    dense_layer_forward(&(conv5->out), dense, 1);

    print_tensor(&(dense->out));
}
