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

    convLayer *conv1 = new_conv_layer(3,   64,  11, 11, 4, 4, 2);
    convLayer *conv2 = new_conv_layer(64,  192, 5,  5,  1, 1, 2);
    convLayer *conv3 = new_conv_layer(192, 384, 3,  3,  1, 1, 1);
    convLayer *conv4 = new_conv_layer(384, 256, 3,  3,  1, 1, 1);
    convLayer *conv5 = new_conv_layer(256, 256, 3,  3,  1, 1, 1);
    poolLayer *pool = new_pool_layer(2, 2, 2, 2, 0, MAXPOOL);
//    poolLayer *pool2 = new_pool_layer(2, 2, 2, 2, 0, AVPOOL);
    denseLayer *dense = new_dense_layer(10, 49 * 256);
    init_conv_layer(conv1);
    init_conv_layer(conv2);
    init_conv_layer(conv3);
    init_conv_layer(conv4);
    init_conv_layer(conv5);
    init_dense_layer(dense);
    convLayer_forward(&(input_layer.out), conv1, 1);
    reluAct_forward(&(conv1->out));
    convLayer_forward(&(conv1->out), conv2, 1);
    reluAct_forward(&(conv2->out));

    poolLayer_forward(&(conv2->out), pool);
//    poolLayer_forward(&(pool->out), pool2);

    convLayer_forward(&(pool->out), conv3, 1);
    reluAct_forward(&(conv3->out));
    convLayer_forward(&(conv3->out), conv4, 1);
    reluAct_forward(&(conv4->out));
    convLayer_forward(&(conv4->out), conv5, 1);
    reluAct_forward(&(conv5->out));
    poolLayer_forward(&(conv5->out), pool);
    denseLayer_forward(&(conv5->out), dense, 1);

    print_tensor(&(dense->out));
}
