#include "./common.cpp"

const unsigned int SIZE = 256 * 1024 * 1024;
const int MAX_THREAD_SIZE_PER_BLOCK = 1024;

// REMIND
// 여기 unsigned int n 으로 안 하고 int n 으로 하면 안 됨!
// SIZE 256M만 되어도 int로 표현 못함.. long, long long, unsinged int 
// 뭐 이런 걸로 해결해야 함..
__global__ void kernelAdd(float* dst, float* src1, float* src2, unsigned int n) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        dst[i] = src1[i] + src2[i];
    }
}

int main(void) {
    float * vecA, *vecB, *vecC;

    vecA = (float*)malloc(SIZE * sizeof(float));
    vecB = (float*)malloc(SIZE * sizeof(float));
    vecC = (float*)malloc(SIZE * sizeof(float));

    srand(42);
    setNormalizedRandomData(vecA, SIZE);
    setNormalizedRandomData(vecB, SIZE);

    float * dev_vecA = nullptr;
    float * dev_vecB = nullptr;
    float * dev_vecC = nullptr;

    cudaMalloc(&dev_vecA, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_vecB, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_vecC, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(1);
    cudaMemcpy(dev_vecA, vecA, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(dev_vecB, vecB, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(MAX_THREAD_SIZE_PER_BLOCK);
    dim3 dimGrid(SIZE + (dimBlock.x - 1) / dimBlock.x);
    ELAPSED_TIME_BEGIN(0);
    kernelAdd<<<dimGrid, dimBlock>>>(dev_vecC, dev_vecA, dev_vecB, SIZE);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(0);
    CUDA_CHECK_ERROR();

    cudaMemcpy(vecC, dev_vecC, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    ELAPSED_TIME_END(1);
    CUDA_CHECK_ERROR();

    cudaFree(dev_vecA);
    CUDA_CHECK_ERROR();
    cudaFree(dev_vecB);
    CUDA_CHECK_ERROR();
    cudaFree(dev_vecC);

    float sumA = getSum(vecA, SIZE);
    float sumB = getSum(vecB, SIZE);
    float sumC = getSum(vecC, SIZE);
    float diff = fabs(sumC - (sumA + sumB));

    printf("SIZE = %d\n", SIZE);
    printf("sumA = %f\n", sumA);
    printf("sumB = %f\n", sumB);
    printf("sumC = %f\n", sumC);
    printf("diff(sumC, sumA+sumB) = %f\n", diff);
    printf("diff(sumC, sumA+sumB) / SIZE = %f\n", diff / SIZE);
    printVec("vecA", vecA, SIZE);
    printVec("vecB", vecB, SIZE);
    printVec("vecC", vecC, SIZE);

    free(vecA);
    free(vecB);
    free(vecC);

    return 0;
}