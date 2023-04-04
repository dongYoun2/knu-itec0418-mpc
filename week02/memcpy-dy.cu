#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

void printArray(const float* arr, int len) {
    for (int i = 0; i < len; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main(void) {
    // host-side data
    const int SIZE = 8;
    const float a[SIZE] = {1., 2., 3., 4., 5., 6., 7., 8.};
    float b[SIZE] = {0., 0., 0., 0., 0., 0., 0., 0.};

    printf("a: ");
    printArray(a, SIZE);

    // device-side data
    float* dev_ptr1 = nullptr;
    float* dev_ptr2 = nullptr;

    // 근데 이렇게 해도 동작은 함
    cudaMalloc(&dev_ptr1, sizeof(a));
    cudaMalloc(&dev_ptr2, sizeof(a));

    cudaMemcpy(dev_ptr1, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr2, dev_ptr1, sizeof(a), cudaMemcpyDeviceToDevice);
    cudaMemcpy(b, dev_ptr2, sizeof(a), cudaMemcpyDeviceToHost);

    // memory release 해주는 거 꼭 해줘야 함!
    cudaFree(dev_ptr1);
    cudaFree(dev_ptr2);

    printf("b: ");
    printArray(b, SIZE);

    return 0;
}