#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

void printArray(const float* arr, int len, char name) {
    printf("%c: ", name);
    for (int i = 0; i < len; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

__global__ void addOne(float* dst, float* src) {
    int i = threadIdx.x;
    // 주의 1.0f 처럼 f 표시 해야함!
    dst[i] = src[i] + 1.0f;
}

int main(void) {
    // host-side data
	const int SIZE = 8;
	const float a[SIZE] = { 0., 1., 2., 3., 4., 5., 6., 7. };
	float b[SIZE] = { 0., 0., 0., 0., 0., 0., 0., 0. };

    printArray(a, SIZE, 'a');

    // device-side data
    float* dev_a = nullptr;
    float* dev_b = nullptr;

    cudaMalloc((void**)&dev_a, SIZE * sizeof(float));
    cudaMalloc((void**)&dev_b, SIZE * sizeof(float));

    cudaMemcpy((void*)dev_a, a, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    addOne<<<1, SIZE>>>(dev_b, dev_a);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess ) {
        printf("CUDA: ERROR: %s\n", cudaGetErrorString(err));
        exit(1);
    } else {
        printf("CUDA: Success\n");
    }

    cudaMemcpy((void*)b, dev_b, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printArray(b, SIZE, 'b');

    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}