#ifndef CIFAR10LOADER_H
#define CIFAR10LOADER_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "Tensor.h"
#include "Utilities.h"

#define CIFAR_IMAGE_W 32
#define CIFAR_IMAGE_H 32
#define CIFAR_CHANNEL 3
#define CIFAR_IMAGE_SIZE CIFAR_IMAGE_W * CIFAR_IMAGE_H * CIFAR_CHANNEL
#define TEST_IMG 100

void cifar10_load(const char *tf, int start, int num, FloatTensor *pixel, FloatTensor *label);

#endif //CIFAR10LOADER_H
