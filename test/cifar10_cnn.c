#include "ESP_RE.h"
#include "Cifar10Loader.h"
#define THRESHOLD 0.1

const char * image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";
void print_tensor(FloatTensor* tensor);
void random_init_arr(float* arr, size_t arr_length);


int main(){
    FloatTensor cifar_image = tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);
    cifar10_load(image_path, 0, 1, &cifar_image, &cifar_label);
    //print_tensor(&cifar_image);



    return 0;
}

void print_tensor(FloatTensor* tensor){
    int count = 0;
    for(int i=0; i<tensor->M; i++){
        for(int j=0; j<tensor->N; j++){
            printf("%.2f, ", tensor->data[i*tensor->M + j]);
            if(tensor->data[i*tensor->M + j] == -1) count++;
        }
        puts("");
    }
}

void random_init_arr(float* arr, size_t arr_length){
    for(int i=0; i<arr_length; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr[i] = 1.0f;
        else arr[i] = 0.0f;
    }
}