#include "BitPackingEspresso/BPTensor.h"


BPTensor bp_tensor_init(int D, int M, int N, int L) {
    size_t packed_size = D * M * N * L;
    if (packed_size % 32 != 0){
//        fprintf(stderr,
//                "Invalid tensor size: %lu. Tensor size should be a multiple of 32.\n", packed_size);
        //exit(-1);
        packed_size = ceil(packed_size / 32) + 1;
    }else
        packed_size /= 32;

    BPTensor t = {D, M, N, L, M * N * L, sizeof(__uint32_t) * packed_size, packed_size};
//    t.data = calloc(t.packed_len, sizeof(__uint32_t));
    t.data = malloc(t.packed_len * sizeof(__uint32_t));
    return t;
}

BPTensor bp_tensor_zeros(int D, int M, int N, int L) {
    BPTensor t = bp_tensor_init(D, M, N, L);
    memset(t.data, 0, t.bytes);
    return t;
}

BPTensor bp_tensor_copy(BPTensor *in) {
    const int D = in->D, M = in->M, N = in->N, L = in->L;
    BPTensor out = bp_tensor_init(D, M, N, L);
    if (!in->data){
        fprintf(stderr, "Tensor Copy Error: Input tensor has no data.\n");
        exit(-1);
    }
    memcpy(out.data, in->data, in->bytes);
    return out;
}

void unpack_bp_tensor(__uint8_t *unpacked_tensor, const BPTensor *tensor){
    int flag = 0;
    if ((tensor->MNL * tensor->D) != (tensor->packed_len * 32))
        flag = 1;

    for (int num_idx = 0; num_idx < tensor->packed_len; num_idx++){
        __uint32_t tmp = tensor->data[num_idx];

        if (flag && num_idx == tensor->packed_len - 1){
            // discard the padded zeros
            int pad_len = tensor->packed_len * 32 - tensor->MNL * tensor->D;
            tmp = tmp>>pad_len;
            for (int i = 31 - pad_len; i >= 0; i--){
                if (tmp & 0x01) unpacked_tensor[num_idx*32+i] = 1;
                else unpacked_tensor[num_idx*32+i] = 0;
                tmp = tmp>>1;
            }
        }else{
            for (int i = 31; i >= 0; i--){
                if (tmp & 0x01) unpacked_tensor[num_idx*32+i] = 1;
                else unpacked_tensor[num_idx*32+i] = 0;
                tmp = tmp>>1;
            }
        }
    }
}

void pack_array_into_bp_tensor(const __uint8_t *arr, BPTensor *tensor){
    int flag = 0;
    if ((tensor->MNL * tensor->D) != (tensor->packed_len * 32))
        flag = 1;

    for (int num_idx = 0; num_idx < tensor->packed_len; num_idx++){
        tensor->data[num_idx] = 0;

        if (flag && num_idx == tensor->packed_len - 1) {
            // discard the padded zeros
            int pad_len = tensor->packed_len * 32 - tensor->MNL * tensor->D;
            tensor->data[num_idx] = tensor->data[num_idx] << pad_len;
            for (int i = 0; i < 32 - pad_len; i++) {
                tensor->data[num_idx] = (tensor->data[num_idx]) | (arr[num_idx * 32 + i] << i);
            }
        }else{
            for (int i = 0; i < 32; i++){
//            __uint32_t aaa = (tensor->data[num_idx]);
//            __uint32_t bbb = arr[num_idx*32+i];
                if (num_idx*32+i > tensor->packed_len * 32)
                    fprintf(stderr, "Array Index %d Outbound!\n", num_idx*32+i);
                tensor->data[num_idx] = (tensor->data[num_idx]) | (arr[num_idx*32+i]<<i);
            }
        }
    }
}

BPTensor bp_tensor_copy_pad(BPTensor *t, int p) {
    const int Ms = t->M, Ns = t->N, L = t->L, D = t->D;
    const int Md = PAD(Ms, p), Nd = PAD(Ns, p);
    BPTensor out = bp_tensor_init(D, Md, Nd, L);
    __uint8_t *pin = malloc(sizeof(__uint8_t) * t->packed_len * 32);
    unpack_bp_tensor(pin, t);
    __uint8_t *pout = calloc(out.packed_len * 32, sizeof(__uint8_t));
    for (int w = 0; w < D; w++) {
        for (int i = 0; i < Ms; i++)
            for (int j = 0; j < Ns; j++)
                for (int k = 0; k < L; k++)
                    pout[ID3(i + p, j + p, k, Nd, L)] =
                            pin[ID3(i, j, k, Ns, L)];
        pin += t->MNL;
        pout += out.MNL;
    }

    pout -= out.MNL * D;
    pin -= t->MNL * D;
    pack_array_into_bp_tensor(pout, &out);
    free(pin);
    free(pout);
    return out;
}

BPTensor bp_tensor_from_ptr(int D, int M, int N, int L, __uint32_t *ptr) {
    BPTensor result = {D, M, N, L, M * N * L, BYTES(__uint32_t , D * M * N * L / 32)};
    ASSERT(ptr, "err: NULL ptr\n");
    result.data = ptr;
    return result;
}

void bp_tensor_tch(BPTensor *a, BPTensor *b) {
    const int M = a->M, N = a->N, L = a->L, D = a->D;
    __uint8_t *unpacked_a = malloc(sizeof(__uint8_t) * a->packed_len * 32);
    unpack_bp_tensor(unpacked_a, a);
    __uint8_t *unpacked_b = malloc(sizeof(__uint8_t) * b->packed_len * 32);
    unpack_bp_tensor(unpacked_b, b);
    for (int w = 0; w < D; w++) {
        __uint8_t *src = unpacked_a + w * a->MNL;
        __uint8_t *dst = unpacked_b + w * b->MNL;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < L; k++)
                    dst[ID3(j, k, i, L, M)] =
                            src[ID3(i, j, k, N, L)];
    }
    pack_array_into_bp_tensor(unpacked_b, b);
    free(unpacked_a);
    free(unpacked_b);
}

void bp_tensor_clear(BPTensor *t){
    memset(t->data, 0, t->bytes);
}

void bp_tensor_pad(BPTensor *src, BPTensor *dst, int p) {
    const int D = src->D, L = src->L;
    const int Ms = src->M, Ns = src->N;
    const int Md = dst->M, Nd = dst->N;
    ASSERT(D == dst->D && L == dst->L, "err: pad shape\n");
    __uint8_t *unpacked_src = malloc(sizeof(__uint8_t) * src->packed_len * 32);
    unpack_bp_tensor(unpacked_src, src);
    __uint8_t *unpacked_dst = malloc(sizeof(__uint8_t) * src->packed_len * 32);
    //unpack_bp_tensor(unpacked_dst, dst);
//    __uint8_t *s = unpacked_src;
//    __uint8_t *d = unpacked_dst;
    memset(unpacked_dst, 0, (dst->D * dst->MNL * sizeof(__uint8_t)));
    for (int w = 0; w < D; w++) {
        for (int i = 0; i < Ms; i++)
            for (int j = 0; j < Ns; j++)
                for (int k = 0; k < L; k++)
                    unpacked_dst[ID3(i + p, j + p, k, Nd, L)] =
                            unpacked_src[ID3(i, j, k, Ns, L)];
        unpacked_src += src->MNL;
        unpacked_dst += dst->MNL;
    }
    //pack_array_into_bp_tensor(s, src);
    unpacked_dst -= dst->MNL;
    pack_array_into_bp_tensor(unpacked_dst, dst);
    free(unpacked_src);
    free(unpacked_dst);
}

void bp_tensor_maxpool(BPTensor *input, BPTensor *output, int pool_kernel_w, int pool_kernel_h,
        int Sx, int Sy) {
    int batch = input->D, kernel_h = input->M, kernel_w = input->N, channel = input->L;
    int output_h = output->M, output_w = output->N;
    int channel_size = input->MNL;
    int input_kernel_size = kernel_h * kernel_w;
    int n = 0;
    int outbound_count = 0, loop_count = 0;

    __uint8_t *unpacked_input = malloc(sizeof(__uint8_t) * input->packed_len * 32);
    unpack_bp_tensor(unpacked_input, input);

//    puts("The packed input of maxpool layer: ");
//    for (int i = 0; i < input->packed_len; i++){
//        //if ((i+1)%32==0) printf(" | ");
//        print_bits(&input->data[i]);
////        __uint32_t num = input->data[i];
////        for (int size = 0; size < 32; size++){
////            num = num<<size;
////            printf()
////        }
//    }
//        print_bits(&input->data[i]);

//    puts("\nThe unpacked input of maxpool layer: ");
//    for (int i = 0; i < input->packed_len * 32; i++){
//        printf("%u,", unpacked_input[i]);
//        if ((i+1)%32==0) printf(" | ");
//    }

    __uint8_t *unpacked_output = malloc(sizeof(__uint8_t) * output->packed_len * 32);
    //unpack_bp_tensor(unpacked_output, output);

    if (input_kernel_size == 1){
        for (int i = 0; i < output->MNL; i++)
            unpacked_output[i] = unpacked_input[i];
    }else{
        for (int idx_b = 0; idx_b < batch; idx_b++){
            for (int idx_c = 0; idx_c < channel; idx_c++){
                int kernel_loc = idx_b * channel_size + idx_c * input_kernel_size;
                for (size_t output_idx_h = 0; output_idx_h < output_h; output_idx_h++){
                    for (size_t output_idx_w = 0; output_idx_w < output_w; output_idx_w++){
                        __uint8_t max = 0;
                        for (size_t i = 0; i < pool_kernel_h; i++){
                            const size_t s = output_idx_h * Sy + i;
                            if (s < kernel_h){
                                for (size_t j = 0; j < pool_kernel_w; j++){
                                    const size_t t = output_idx_w * Sx + j;
                                    loop_count++;
                                    if(kernel_loc + s + t >= (input->MNL * input->D)){
                                        outbound_count++;
                                        fprintf(stderr, "Input Maxpool: Index outbund!\n");
                                    }
                                    if (t < kernel_w)
                                        if (unpacked_input[kernel_loc + s + t] > 0) max = 1;
                                }
                            }
                        }
                        unpacked_output[n] = max;
                        if (n >= (output->MNL * output->D)) {
                            outbound_count++;
                            fprintf(stderr, "Output Maxpool: Index outbund!\n");
                        }
                        n++;
                    }
                }
            }
        }
    }

    pack_array_into_bp_tensor(unpacked_output, output);
    free(unpacked_input);
    free(unpacked_output);
}

void bp_tensor_avgpool(BPTensor *input, BPTensor *output, int pool_kernel_w, int pool_kernel_h,
        int Sx, int Sy) {
    int pool_kernel_size = pool_kernel_w * pool_kernel_h;
    int batch = input->D, kernel_h = input->M, kernel_w = input->N, channel = input->L;
    int output_h = output->M, output_w = output->N;
    int channel_size = input->MNL;
    int input_kernel_size = kernel_h * kernel_w;
    int n = 0;
    int outbound_count = 0, loop_count = 0;

    __uint8_t *unpacked_input = malloc(sizeof(__uint8_t) * input->packed_len * 32);
    unpack_bp_tensor(unpacked_input, input);
    __uint8_t *unpacked_output = malloc(sizeof(__uint8_t) * output->packed_len * 32);
    //unpack_bp_tensor(unpacked_output, output);

    if (input_kernel_size == 1){
        for (int i = 0; i < output->MNL; i++)
            unpacked_output[i] = unpacked_input[i];
    }else{
        for (int idx_b = 0; idx_b < batch; idx_b++){
            for (int idx_c = 0; idx_c < channel; idx_c++){
                int kernel_loc = idx_b * channel_size + idx_c * input_kernel_size;
                for (size_t output_idx_h = 0; output_idx_h < output_h; output_idx_h++){
                    for (size_t output_idx_w = 0; output_idx_w < output_w; output_idx_w++){
                        __uint8_t sum = 0;
                        for (size_t i = 0; i < pool_kernel_h; i++){
                            const size_t s = output_idx_h * Sy + i;
                            if (s < kernel_h){
                                for (size_t j = 0; j < pool_kernel_w; j++){
                                    const size_t t = output_idx_w * Sx + j;
                                    loop_count++;
                                    if(kernel_loc + s + t >= (input->MNL * input->D)){
                                        outbound_count++;
                                        fprintf(stderr, "Avgpool: Index outbund!\n");
                                    }
                                    if (t < kernel_w){
                                        sum += unpacked_input[kernel_loc + s + t];
                                    }
                                }
                            }
                        }
                        unpacked_output[n] = sum / pool_kernel_size;
                        n++;
                    }
                }
            }
        }
    }

    pack_array_into_bp_tensor(unpacked_output, output);
    free(unpacked_input);
    free(unpacked_output);
}

void bp_tensor_lower(BPTensor *input, BPTensor *output,
        int conv_kernel_w, int conv_kernel_h, int Sx, int Sy) {
    const int D = input->D;
    const int Ms = input->M, Ns = input->N, Ls = input->L;
    const int Md = output->M, Nd = output->N, Ld = output->L;
    ASSERT(output->D == D, "err: lowering shape\n")

    __uint8_t *unpacked_input = malloc(sizeof(__uint8_t) * input->packed_len * 32);
    unpack_bp_tensor(unpacked_input, input);
    __uint8_t *unpacked_output = malloc(sizeof(__uint8_t) * output->packed_len * 32);
    //unpack_bp_tensor(unpacked_output, output);

    __uint8_t *d = unpacked_output;
    int n = 0;
    int outbound_count = 0, loop_count = 0;
    for (int w = 0; w < D; w++) {
        __uint8_t *s = unpacked_input + w * input->MNL;
        for (int i = 0; i < Md; i++) {
            for (int j = 0; j < Nd; j++) {
                for (int y = 0; y < conv_kernel_h; y++) {
                    for (int x = 0; x < conv_kernel_w; x++) {
                        for (int k = 0; k < Ls; k++) {
                            int id3 = ID3(i * Sy + y, j * Sx + x, k, Ns, Ls);
                            loop_count++;
                            if (id3 >= (input->MNL * input->D)) {
                                outbound_count++;
                                fprintf(stderr, "Source Tensor Lower: Index outbund!\n");
                            }
                            d[n] = s[id3];
                            if (n >= (output->MNL * output->D)) {
                                outbound_count++;
                                fprintf(stderr, "Destiny Tensor Lower: Index outbund!\n");
                            }
                            n++;
                        }
                    }
                }
            }
        }
    }
    pack_array_into_bp_tensor(unpacked_output, output);
//    puts("The outputs of lowered tensor: ");
//    for (int i = 0; i < output->packed_len; i++)
//        printf("%u, ", output->data[i]);
//    puts("");
    free(unpacked_input);
    free(unpacked_output);
}

void bp_tensor_free(BPTensor *t) {
    if (t->data) {
        free(t->data);
    }
}

void bp_print_tensor(BPTensor *tensor) {
    __uint8_t *unpacked_tensor = malloc(sizeof(__uint8_t) * tensor->packed_len * 32);
    unpack_bp_tensor(unpacked_tensor, tensor);
    printf("The shape of the tensor is: %d x %d x %d x %d\n", tensor->D, tensor->M, tensor->N, tensor->L);
    for (int ch = 0; ch < tensor->L; ch++){
        for (int i = 0; i < tensor->M; i++) {
            for (int j = 0; j < tensor->N; j++) {
                printf("%u, ", unpacked_tensor[ch * tensor->M * tensor->N + i * tensor->M + j]);
            }
            puts("");
        }
    }
    free(unpacked_tensor);
}


void bp_unpack_to_float(float *arr_float, const __uint32_t *arr_packed, size_t packed_size) {
    for (int num_idx = 0; num_idx < packed_size; num_idx++){
        __uint32_t tmp = arr_packed[num_idx];
        for (int i = 31; i >= 0; i--){
            if (tmp & 0x01) arr_float[num_idx*32+i] = 1.0f;
            else arr_float[num_idx*32+i] = 0.0f;
            tmp = tmp>>1;
        }
    }
}

void bp_pack_from_float(const float *arr_float, __uint32_t *arr_packed, size_t packed_size) {
    for (int num_idx = 0; num_idx < packed_size; num_idx++){
        arr_packed[num_idx] = 0;
        for (int i = 0; i < 32; i++){
            __uint8_t tmp = 0;
            if(arr_float[num_idx*32+i] > 0) tmp = 1;
            arr_packed[num_idx] =
                    (arr_packed[num_idx]) | (tmp<<i);
        }
    }
}

void bp_tensor_cat(BPTensor *tensor_a, BPTensor *tensor_b, BPTensor *result, int dim) {
    if(dim == 0){
        if((tensor_a->M != tensor_b->M)
           || (tensor_a->N != tensor_b->N)
           || (tensor_a->L != tensor_b->L))
            fprintf(stderr, "Dimension doesn't match\n");
        result->D = tensor_a->D + tensor_b->D;
        result->M = tensor_a->M;
        result->N = tensor_a->N;
        result->L = tensor_a->L;
        result->MNL = tensor_a->MNL;
    }else if(dim == 1){
        if((tensor_a->D != tensor_b->D)
           || (tensor_a->N != tensor_b->N)
           || (tensor_a->L != tensor_b->L))
            fprintf(stderr, "Dimension doesn't match\n");
        result->D = tensor_a->D;
        result->M = tensor_a->M + tensor_b->M;
        result->N = tensor_a->N;
        result->L = tensor_a->L;
        result->MNL = tensor_a->MNL;
    }else if(dim == 2){
        if((tensor_a->D != tensor_b->D)
           || (tensor_a->M != tensor_b->M)
           || (tensor_a->L != tensor_b->L))
            fprintf(stderr, "Dimension doesn't match\n");
        result->D = tensor_a->D;
        result->M = tensor_a->M;
        result->N = tensor_a->N + tensor_b->N;
        result->L = tensor_a->L;
        result->MNL = tensor_a->MNL;
    }else if(dim == 3){
        if((tensor_a->D != tensor_b->D)
           || (tensor_a->N != tensor_b->N)
           || (tensor_a->M != tensor_b->M))
            fprintf(stderr, "Dimension doesn't match\n");
        result->D = tensor_a->D;
        result->M = tensor_a->M;
        result->N = tensor_a->N;
        result->L = tensor_a->L + tensor_b->L;
        result->MNL = result->M * result->N * result->L;
    }

    result->bytes = BYTES(__uint32_t , result->packed_len);
    result->packed_len = tensor_a->packed_len + tensor_b->packed_len;
    result->data = malloc(result->packed_len * sizeof(__uint32_t));

    for(int idx = 0; idx < tensor_a->packed_len; idx++){
        result->data[idx] = tensor_a->data[idx];
    }
    for(int idx = 0; idx < tensor_b->packed_len; idx++){
        result->data[tensor_a->packed_len + idx] = tensor_b->data[idx];
    }
}
