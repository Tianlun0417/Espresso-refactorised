#include "FloatTypeEspresso/FLOAT_ESP.h"
#include "FloatTypeEspresso/Cifar10Loader.h"
#include <stdbool.h>

const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";
bool use_batch_norm = false;

int main() {
    FloatTensor cifar_image = tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);

    InputLayer input_layer;

    ConvLayer *conv_layer_1 = new_conv_layer(3,   64,  3, 3, 1, 1, 1);
    ConvLayer *conv_layer_2 = new_conv_layer(64,  128, 3, 3, 1, 1, 1);
    ConvLayer *conv_layer_3 = new_conv_layer(128, 256, 3, 3, 1, 1, 1);
    ConvLayer *conv_layer_4 = new_conv_layer(256, 256, 3, 3, 1, 1, 1);
    ConvLayer *conv_layer_5 = new_conv_layer(256, 512, 3, 3, 1, 1, 1);
    ConvLayer *conv_layer_6 = new_conv_layer(512, 512, 3, 3, 1, 1, 1);
    ConvLayer *conv_layer_7 = new_conv_layer(512, 512, 3, 3, 1, 1, 1);
    ConvLayer *conv_layer_8 = new_conv_layer(512, 512, 3, 3, 1, 1, 1);

    PoolLayer maxPool_layer = poolLayer_init(2, 2, 2, 2, MAXPOOL);

    DropoutLayer *dropout_layer = new_dropout_layer(0.5);

    DenseLayer dense_layer_1 = denseLayer_init(4096, 512);
    DenseLayer dense_layer_2 = denseLayer_init(4096, 4096);
    DenseLayer dense_layer_3 = denseLayer_init(10, 4096);

    bnormLayer bnorm_layer_1 = bnormLayer_init(0);
    bnormLayer bnorm_layer_2 = bnormLayer_init(0);
    bnormLayer bnorm_layer_3 = bnormLayer_init(0);
    bnormLayer bnorm_layer_4 = bnormLayer_init(0);
    bnormLayer bnorm_layer_5 = bnormLayer_init(0);
    bnormLayer bnorm_layer_6 = bnormLayer_init(0);
    bnormLayer bnorm_layer_7 = bnormLayer_init(0);
    bnormLayer bnorm_layer_8 = bnormLayer_init(0);

    /*------------------------Initialise each layer------------------------*/
    init_conv_layer(conv_layer_1);
    init_conv_layer(conv_layer_2);
    init_conv_layer(conv_layer_3);
    init_conv_layer(conv_layer_4);
    init_conv_layer(conv_layer_5);
    init_conv_layer(conv_layer_6);
    init_conv_layer(conv_layer_7);
    init_conv_layer(conv_layer_8);

    init_dense_layer(&dense_layer_1);
    init_dense_layer(&dense_layer_2);
    init_dense_layer(&dense_layer_3);

    if (use_batch_norm) {
        init_batchnorm_layer(&bnorm_layer_1, 64);
        init_batchnorm_layer(&bnorm_layer_2, 128);
        init_batchnorm_layer(&bnorm_layer_3, 256);
        init_batchnorm_layer(&bnorm_layer_4, 256);
        init_batchnorm_layer(&bnorm_layer_5, 512);
        init_batchnorm_layer(&bnorm_layer_6, 512);
        init_batchnorm_layer(&bnorm_layer_7, 512);
        init_batchnorm_layer(&bnorm_layer_8, 512);
    }
    /*------------------------Initialise each layer------------------------*/

    int save = 0;

    for (int idx = 0; idx < 1; idx++) {
        cifar10_load(image_path, idx, 1, &cifar_image, &cifar_label);

        input_layer_load(&cifar_image, &input_layer);
        input_layer_forward(&input_layer);

        conv_layer_forward(&input_layer.out, conv_layer_1, save);
        if (use_batch_norm) {
            bnormLayer_forward(&conv_layer_1->out, &bnorm_layer_1, save);
        }
        relu_forward(&conv_layer_1->out);

        pool_layer_forward(&conv_layer_1->out, &maxPool_layer);
        conv_layer_forward(&maxPool_layer.out, conv_layer_2, save);
        tensor_free(&maxPool_layer.out);
        if (use_batch_norm) {
            bnormLayer_forward(&conv_layer_2->out, &bnorm_layer_2, save);
        }
        relu_forward(&conv_layer_2->out);

        pool_layer_forward(&conv_layer_2->out, &maxPool_layer);
        conv_layer_forward(&maxPool_layer.out, conv_layer_3, save);
        tensor_free(&maxPool_layer.out);
        if (use_batch_norm) {
            bnormLayer_forward(&conv_layer_3->out, &bnorm_layer_3, save);
        }
        relu_forward(&conv_layer_3->out);

        conv_layer_forward(&conv_layer_3->out, conv_layer_4, save);
        if (use_batch_norm) {
            bnormLayer_forward(&conv_layer_4->out, &bnorm_layer_4, save);
        }
        relu_forward(&conv_layer_4->out);

        pool_layer_forward(&conv_layer_4->out, &maxPool_layer);
        conv_layer_forward(&maxPool_layer.out, conv_layer_5, save);
        tensor_free(&maxPool_layer.out);
        if (use_batch_norm) {
            bnormLayer_forward(&conv_layer_5->out, &bnorm_layer_5, save);
        }
        relu_forward(&conv_layer_5->out);

        conv_layer_forward(&conv_layer_5->out, conv_layer_6, save);
        if (use_batch_norm) {
            bnormLayer_forward(&conv_layer_6->out, &bnorm_layer_6, save);
        }
        relu_forward(&conv_layer_6->out);

        pool_layer_forward(&conv_layer_6->out, &maxPool_layer);
        conv_layer_forward(&maxPool_layer.out, conv_layer_7, save);
        tensor_free(&maxPool_layer.out);
        if (use_batch_norm) {
            bnormLayer_forward(&conv_layer_7->out, &bnorm_layer_7, save);
        }
        relu_forward(&conv_layer_7->out);

        conv_layer_forward(&conv_layer_7->out, conv_layer_8, save);
        if (use_batch_norm) {
            bnormLayer_forward(&conv_layer_8->out, &bnorm_layer_8, save);
        }
        relu_forward(&conv_layer_8->out);
        pool_layer_forward(&conv_layer_8->out, &maxPool_layer);

        dense_layer_forward(&maxPool_layer.out, &dense_layer_1, save);
        relu_forward(&dense_layer_1.out);
        dropoutLayer_forward(&dense_layer_1.out, dropout_layer);
        dense_layer_forward(&dense_layer_1.out, &dense_layer_2, save);
        relu_forward(&dense_layer_2.out);
        dropoutLayer_forward(&dense_layer_2.out, dropout_layer);
        dense_layer_forward(&dense_layer_2.out, &dense_layer_3, save);

        printf("Test img NO.%d ", idx);
        print_tensor(&dense_layer_3.out);

        tensor_free(&input_layer.out);

        tensor_free(&conv_layer_1->in);
        tensor_free(&conv_layer_1->out);
        tensor_free(&conv_layer_2->in);
        tensor_free(&conv_layer_2->out);
        tensor_free(&conv_layer_3->in);
        tensor_free(&conv_layer_3->out);
        tensor_free(&conv_layer_4->in);
        tensor_free(&conv_layer_4->out);
        tensor_free(&conv_layer_5->in);
        tensor_free(&conv_layer_5->out);
        tensor_free(&conv_layer_6->in);
        tensor_free(&conv_layer_6->out);
        tensor_free(&conv_layer_7->in);
        tensor_free(&conv_layer_7->out);
        tensor_free(&conv_layer_8->in);
        tensor_free(&conv_layer_8->out);

        tensor_free(&dense_layer_1.in);
        tensor_free(&dense_layer_1.out);
        tensor_free(&dense_layer_2.in);
        tensor_free(&dense_layer_2.out);
        tensor_free(&dense_layer_3.in);
        tensor_free(&dense_layer_3.out);

        tensor_free(&maxPool_layer.out);
    }

    return 0;
}
