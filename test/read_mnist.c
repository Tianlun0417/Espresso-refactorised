#include "MnistLoader.h"
#include <stdio.h>


const char * image_path = "/home/tianlun/codes/espresso-refactorised/data/t10k-images-idx3-ubyte";
const char * label_path = "/home/tianlun/codes/espresso-refactorised/data/t10k-labels-idx1-ubyte";

void print_mnist_image(int* image_array);

int main(){
    mnist_dataset_t * mnist_dataset = mnist_get_dataset(image_path, label_path);
    for(int loop=0; loop<10000; loop++) {
        int image_tmp[784];
        for (int i = 0; i < 784; i++) {
            image_tmp[i] = mnist_dataset->images->pixels[i];
        }

        uint8_t label = *(mnist_dataset->labels);
        printf("\nThe actual digit should be %u.\n", label);

        print_mnist_image(&image_tmp[0]);

        mnist_dataset->images++;
        mnist_dataset->labels++;
    }
    return 0;
}

void print_mnist_image(int* image_array){
    for(int i=0; i<MNIST_IMAGE_HEIGHT; i++){
        for(int j=0; j<MNIST_IMAGE_WIDTH; j++){
            if(image_array[i*MNIST_IMAGE_WIDTH + j]>=1) printf("%d", 1);
            else printf("%d", 0);
        }
        puts("");
    }
    puts("");
}