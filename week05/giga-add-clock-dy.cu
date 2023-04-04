#include "./common.cpp"

const unsigned int SIZE = 256 * 1024 * 1024;
const int MAX_THREAD_SIZE_PER_BLOCK = 1024;

// REMIND
// 여기서 계산되는 clock_t 형은 clock이 몇 번 튀었는지
// 나타내는 clock tick count 라 그냥 항상 long long 형으로
// 바꿔주는 게 편함!
__global__ void kernelAdd(float* dst, float* src1, float* src2, unsigned int n, long long* times) {
    clock_t start = clock();
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        dst[i] = src1[i] + src2[i];
    }
    clock_t end = clock();

    if(i==0) {
        times[i] = (long long)(end - start);
    }
}

int main(void) {
    float *vecA, *vecB, *vecC;
    long long* host_times;

    vecA = (float*)malloc(SIZE * sizeof(float));
    vecB = (float*)malloc(SIZE * sizeof(float));
    vecC = (float*)malloc(SIZE * sizeof(float));
    host_times = (long long*)malloc(1 * sizeof(long long));
    

    srand(42);
    setNormalizedRandomData(vecA, SIZE);
    setNormalizedRandomData(vecB, SIZE);

    float* dev_vecA = nullptr;
    float* dev_vecB = nullptr;
    float* dev_vecC = nullptr;
    long long* dev_times = nullptr;

    cudaMalloc(&dev_vecA, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_vecB, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_vecC, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_times, 1 * sizeof(long long));

    ELAPSED_TIME_BEGIN(1);
    cudaMemcpy(dev_vecA, vecA, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(dev_vecB, vecB, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(MAX_THREAD_SIZE_PER_BLOCK);
    dim3 dimGrid(SIZE + (dimBlock.x - 1) / dimBlock.x);
    ELAPSED_TIME_BEGIN(0);
    kernelAdd<<<dimGrid, dimBlock>>>(dev_vecC, dev_vecA, dev_vecB, SIZE, dev_times);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(0);
    CUDA_CHECK_ERROR();

    cudaMemcpy(vecC, dev_vecC, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_times, dev_times, 1 * sizeof(long long), cudaMemcpyDeviceToHost);
    ELAPSED_TIME_END(1);
    CUDA_CHECK_ERROR();

    cudaFree(dev_vecA);
    CUDA_CHECK_ERROR();
    cudaFree(dev_vecB);
    CUDA_CHECK_ERROR();
    cudaFree(dev_vecC);
    CUDA_CHECK_ERROR();
    cudaFree(dev_times);
    CUDA_CHECK_ERROR();

    float sumA = getSum(vecA, SIZE);
    float sumB = getSum(vecB, SIZE);
    float sumC = getSum(vecC, SIZE);
    float diff = fabs(sumC - (sumA + sumB));
    
    int gpuClockRate = 1;
    cudaDeviceGetAttribute(&gpuClockRate, cudaDevAttrClockRate, 0);
    float elapsedTime_usec = host_times[0] * 1000.0f / (float)gpuClockRate;
    printf("num clock = %lld, peak clock rate(clock freq.) = %dkHz, elapsed time: %f usec\n",
            host_times[0], gpuClockRate, elapsedTime_usec);
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
    free(host_times);

    return 0;
}