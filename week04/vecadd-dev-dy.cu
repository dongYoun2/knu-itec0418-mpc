#include "./common.cpp"

__global__ void kernel_add(float *dst, float* src1, float* src2, unsigned int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) {
        dst[i] = src1[i] + src2[i];
    }
}

int main(void) {
    const unsigned int SIZE = 1024 * 1024;
    const int MAX_THREAD_NUM = 1024;
    float *a = nullptr;    
    float *b = nullptr;    
    float *c = nullptr;    

    a = (float*)malloc(SIZE * sizeof(float));
    b = (float*)malloc(SIZE * sizeof(float));
    c = (float*)malloc(SIZE * sizeof(float));

    srand(42);
    setNormalizedRandomData(a, SIZE);
    setNormalizedRandomData(b, SIZE);

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;

    ELAPSED_TIME_BEGIN(1);
    cudaMalloc(&dev_a, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_b, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_c, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();

    cudaMemcpy(dev_a, a, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(dev_b, b, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimGrid(SIZE / MAX_THREAD_NUM);
    dim3 dimBlock(MAX_THREAD_NUM);
    ELAPSED_TIME_BEGIN(0);
    kernel_add<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, SIZE);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(0);
    CUDA_CHECK_ERROR();

    cudaMemcpy(c, dev_c, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    ELAPSED_TIME_END(1);

    cudaFree(dev_a);
    CUDA_CHECK_ERROR();
    cudaFree(dev_b);
    CUDA_CHECK_ERROR();
    cudaFree(dev_c);
    CUDA_CHECK_ERROR();

    float sumA = getSum(a, SIZE);
    float sumB = getSum(b, SIZE);
    float sumC = getSum(c, SIZE);
    float diff = fabsf((sumA + sumB) - sumC);

    printf("sum of vector a: %f\n", sumA);
    printf("sum of vector b: %f\n", sumB);

    printf("diff(sumC, sumA+sumB): %f\n", diff);
    printf("mean diff: %f\n", diff / float(SIZE));

    printVec("vecotr a", a, SIZE);
    printVec("vecotr b", b, SIZE);
    printVec("vecotr c", c, SIZE);

    free(a);
    free(b);
    free(c);

    return 0;
}