#include "FloatTypeEspresso/FLOAT_ESP.h"
#include "BitPackingEspresso/BP_ESP.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define _POSIX_C_SOURCE 200809L
#define NUM_TEST 10000
void evaluate_conv_layer();
void evaluate_bp_conv_layer();
void evaluate_dense_layer();
void evaluate_bp_dense_layer();


int main(){
    evaluate_conv_layer();
    evaluate_bp_conv_layer();
    evaluate_dense_layer();
    evaluate_bp_dense_layer();

    return 0;
}

void evaluate_conv_layer(){
    struct timespec tstart={0,0}, tend={0,0};

    FloatTensor test_input = tensor_init(1, 32, 32, 3);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);

    cifar10_load(image_path, 0, 1, &test_input, &cifar_label);

    ConvLayer *new_conv = malloc(sizeof(ConvLayer));
    conv_layer_init(new_conv, 3, 64, 11, 11, 4, 4, 2);
    conv_layer_rand_weight(new_conv);

    puts("For float version convolutional layer: ");
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    for(int i = 0; i < NUM_TEST; i++)
        conv_layer_forward(&test_input, new_conv, 1);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("forward passing took about %.8f seconds\n",
           ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) -
           ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));

    tensor_free(&test_input);
    tensor_free(&cifar_label);
    conv_layer_free(new_conv);
    free(new_conv);
}

void evaluate_bp_conv_layer(){
    struct timespec tstart={0,0}, tend={0,0};

    __uint8_t *bp_test_input = malloc(CIFAR_IMAGE_SIZE * sizeof(__uint8_t));
    __uint8_t *label = malloc(sizeof(__uint8_t));

    cifar10_load_int(image_path, 0, 1, bp_test_input, label);
    BPTensor image_tensor = bp_tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    bp_input_layer_forward(bp_test_input, &image_tensor);

    BPConvLayer *new_bp_conv = malloc(sizeof(BPConvLayer));
    bp_conv_layer_init(new_bp_conv, 3, 64, 11, 11, 4, 4, 2);
    bp_conv_layer_rand_weight(new_bp_conv);

    puts("For bit-packing version convolutional layer: ");
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    for(int i = 0; i < NUM_TEST; i++)
        bp_conv_layer_forward(&image_tensor, new_bp_conv, 1);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("forward passing took about %.8f seconds\n",
           ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) -
           ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));

    bp_tensor_free(&image_tensor);
    bp_conv_layer_free(new_bp_conv);
    free(bp_test_input);
    free(new_bp_conv);
    free(label);
}

void evaluate_dense_layer(){
    struct timespec tstart={0,0}, tend={0,0};
    FloatTensor test_input = tensor_init(1, 1, 1, 256);
    random_init_arr(test_input.data, 256);

    DenseLayer *new_dense_layer = malloc(sizeof(DenseLayer));
    dense_layer_init(new_dense_layer, 4096, 256);
    dense_layer_rand_weight(new_dense_layer);

    puts("For bit-packing version dense layer: ");
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    for(int i = 0; i < NUM_TEST; i++)
        dense_layer_forward(&test_input, new_dense_layer, 1);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("forward passing took about %.8f seconds\n",
           ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) -
           ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));

    tensor_free(&test_input);
    dense_layer_free(new_dense_layer);
    free(new_dense_layer);
}

void evaluate_bp_dense_layer(){
    struct timespec tstart={0,0}, tend={0,0};
    BPTensor test_input = bp_tensor_init(1, 1, 1, 256);
    bp_random_init_packed_arr(test_input.data, 8);

    BPDenseLayer *new_dense_layer = malloc(sizeof(BPDenseLayer));
    bp_dense_layer_init(new_dense_layer, 4096, 256);
    bp_dense_layer_rand_weight(new_dense_layer);

    puts("For bit-packing version dense layer: ");
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    for(int i = 0; i < NUM_TEST; i++)
        bp_dense_layer_forward(&test_input, new_dense_layer, 1);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("forward passing took about %.8f seconds\n",
           ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) -
           ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));

    bp_tensor_free(&test_input);
    bp_dense_layer_free(new_dense_layer);
    free(new_dense_layer);
}