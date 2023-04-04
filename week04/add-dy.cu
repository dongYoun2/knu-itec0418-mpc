#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "./common.cpp"

void printArray(const int* arr, int len, char name) {
    printf("%c: ", name);
    for (int i = 0; i < len; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

__global__ void kernelAdd(int* c, int* a, int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(void) {
    const int SIZE = 5;
    const int a[SIZE] = {1,2,3,4,5};
    const int b[SIZE] = {5,4,3,2,1};
    int c[SIZE] = {0};

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    // REMIND
    // 원래는 이렇게 manually 다 해줘야하지만 매크로 사용하면 됨!
    // 매크로도 사실 매 cuda함수 호출할 때마다 바로 체크해주는 게 좋지만
    // 단순히 에러 발생했는데 계속 실행된 경우가 있는지만 체크해줘도 괜찮음
    // cudaError_t err = cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
    // if(err != cudaSuccess) {
    //     printf("CUDA: ERROR: %s\n", cudaGetErrorString(err));
    //     exit(1);
    // }

    // err = cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
    // if(err != cudaSuccess) {
    //     printf("CUDA: ERROR: %s\n", cudaGetErrorString(err));
    //     exit(1);
    // }

    // err = cudaMalloc((void**)&dev_c, SIZE * sizeof(int));
    // if(err != cudaSuccess) {
    //     printf("CUDA: ERROR: %s\n", cudaGetErrorString(err));
    //     exit(1);
    // }

    // err = cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    // if(err != cudaSuccess) {
    //     printf("CUDA: ERROR: %s\n", cudaGetErrorString(err));
    //     exit(1);
    // }

    // err = cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    // if(err != cudaSuccess) {
    //     printf("CUDA: ERROR: %s\n", cudaGetErrorString(err));
    //     exit(1);
    // }

    cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
    CUDA_CHECK_ERROR();

    cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
    CUDA_CHECK_ERROR();

    cudaMalloc((void**)&dev_c, SIZE * sizeof(int));
    CUDA_CHECK_ERROR();

    cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();


    // REMIND
    // 아래는 되지만
    // for(int i=0; i<SIZE; i++) {
    //     printf("%p ", &dev_a[i]);
    // }

    // REMIND
    // 아래는 안 됨. dev_a가 가리키는 주소는 사실 device side
    // mem. address 이니 그냥 printf 하면 main mem. 주소인지 알고
    // main mem. 에서의 해당 번지에 들어있는 value를 뽑으려 함..!
    // for(int i=0; i<SIZE; i++) {
    //     printf("%d ", dev_a[i]);
    // }

    // REMIND cf.)
    // dim3 a = dim3(1, 1); // == dim3 a(1, 1);

    dim3 gridDim(1);
    dim3 gridBlock(SIZE);
    kernelAdd<<<gridDim, gridBlock>>>(dev_c, dev_a, dev_b);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();

    cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();

    printArray(a, SIZE, 'a');
    printArray(b, SIZE, 'b');
    printArray(c, SIZE, 'c');

    cudaFree(dev_a);
    CUDA_CHECK_ERROR();
    cudaFree(dev_b);
    CUDA_CHECK_ERROR();
    cudaFree(dev_c);
    CUDA_CHECK_ERROR();

    return 0;
}
