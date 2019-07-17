#include "CNN/ResNet.h"


ResNet ResNet_init(enum BlockType block_type, int num_layers[4], int num_classes){
    ResNet ResNetInstance;
    ResNetInstance.inplanes = 64;
    ResNetInstance.conv1 = new_conv_layer(64, 7, 7, 3, 2, 2, 3);
    ResNetInstance.bn1   = new_bn_layer();

}


ResNetBlock* build_ResNet_block(enum BlockType block_type, int planes, int blocks, int stride){

}


convLayer* new_conv_layer(int D, int M, int N, int L,
        int Stride_m, int Stride_n, int padding){
    convLayer* conv_layer_ptr = malloc(sizeof(convLayer));
    conv_layer_ptr->D = D;
    conv_layer_ptr->M = M;
    conv_layer_ptr->N = N;
    conv_layer_ptr->L = L;
    conv_layer_ptr->Stride_m = Stride_m;
    conv_layer_ptr->Stride_n = Stride_n;
    conv_layer_ptr->padding  = padding;
    conv_layer_ptr->W.data   = NULL;
    conv_layer_ptr->b.data   = NULL;
    conv_layer_ptr->in.data  = NULL;
    conv_layer_ptr->out.data = NULL;

    return conv_layer_ptr;
}


bnormLayer* new_bn_layer(){
    bnormLayer* bn_layer_ptr = malloc(sizeof(bnormLayer));
    bn_layer_ptr->N=0;
    bn_layer_ptr->ug=0;
    bn_layer_ptr->mean.data  = NULL;
    bn_layer_ptr->istd.data  = NULL;
    bn_layer_ptr->beta.data  = NULL;
    bn_layer_ptr->gamma.data = NULL;
    bn_layer_ptr->in.data    = NULL;

    return bn_layer_ptr;
}

poolLayer* new_pool_layer(int M, int N, int Stride_m,
                          int Stride_n, poolingStrategy strategy){
    poolLayer* pool_layer_ptr = malloc(sizeof(poolLayer));
    pool_layer_ptr->M = M;
    pool_layer_ptr->N = N;
    pool_layer_ptr->Stride_m  = Stride_m;
    pool_layer_ptr->Stride_n  = Stride_n;
    pool_layer_ptr->strategy  = strategy;
    pool_layer_ptr->out.data  = NULL;
    pool_layer_ptr->mask.data = NULL;

    return pool_layer_ptr;
}
