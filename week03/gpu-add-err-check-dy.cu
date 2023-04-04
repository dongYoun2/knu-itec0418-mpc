#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

void printArray(const int* arr, int len, char name) {
    printf("%c: ", name);
    for (int i = 0; i < len; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

__global__ void kernelAddition(const int* src1, const int* src2, int* dst) {
    int i = threadIdx.x;
    dst[i] = src1[i] + src2[i];
}

int main(void) {
    const int SIZE = 5;
    // host-side data
    const int a[SIZE] = {1,2,3,4,5};
    const int b[SIZE] = {10, 20, 30, 40, 50};
    int c[SIZE] = {0};

    printArray(a, SIZE, 'a');
    printArray(b, SIZE, 'b');

    // device-side data
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c, SIZE * sizeof(int));

    cudaMemcpy((void*)dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    kernelAddition<<<1, SIZE>>>(dev_a, dev_b, dev_c);
    // 아래 cudaDeviceSynchronize(); 꼭 있어야 함! 그래야 모든 커널이 다 
    // 돌아가서 dev_c에 우리가 원하는 게 들어있다는 걸 보장할 수 있음! 
    cudaDeviceSynchronize(); 
    // CUDA 커널 함수 실행 후 에러 체크는 cudaPeekAtLastError() 혹은
    // cudaGetLastError() 로 수행!
    // 꼭 cudaDeviceSynchronize() 까지 완료 후 에러 체크 수행!
    cudaError_t err = cudaPeekAtLastError();
    if(err != cudaSuccess) {
        printf("CUDA: ERROR: cuda failure \"%s\"\n", cudaGetErrorString(err));
        exit(1);
    } else {
        printf("CUDA: Success\n");
    }


    cudaMemcpy((void*)c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // memory release 까먹지 말자! 
    // 그냥 cudaMalloc() 이나 malloc() 쓰면 무조건 일단 바로 다음에 release
    // 하는 함수 적어주자..
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    printArray(c, SIZE, 'c');

    return 0;
}