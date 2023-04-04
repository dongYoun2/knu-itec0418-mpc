#include "./common.cpp"

unsigned nrow = 10000; // num rows
unsigned ncol = 10000; // num columns

__global__ void kernelAdd(float* dst, float* src1, float* src2, size_t dev_pitch, int nrow, int ncol) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < ncol && row < nrow) {
        size_t offset = (row * dev_pitch) + (col * sizeof(float));
        *(float*)((char*)dst + offset) = *(float*)((char*)src1 + offset) + *(float*)((char*)src2 + offset);
    }
}

int main(const int argc, const char* argv[]) {
    switch(argc) {
    case 1:
        break;
    case 2:
        nrow = ncol = procArg<int>(argv[0], argv[1], 32, 16*1024);
        break;
    case 3:
        nrow = procArg<int>(argv[0], argv[1], 32, 16*1024);
        ncol = procArg<int>(argv[0], argv[2], 32, 16*1024);
        break;
    default:
        printf("usage: %s [nrow] [ncol]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // host-side data
    float *matA = (float*)malloc(sizeof(float) * nrow * ncol);
    float *matB = (float*)malloc(sizeof(float) * nrow * ncol);
    float *matC = (float*)malloc(sizeof(float) * nrow * ncol);

    srand(42);
    setNormalizedRandomData(matA, nrow * ncol);
    setNormalizedRandomData(matB, nrow * ncol);

    // device-side data
    float *dev_matA = nullptr;
    float *dev_matB = nullptr;
    float *dev_matC = nullptr;
    size_t dev_pitch;
    size_t host_pitch = (size_t)(ncol * sizeof(float));

    ELAPSED_TIME_BEGIN(1);
    cudaMallocPitch(&dev_matA, &dev_pitch, ncol * sizeof(float), nrow);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matB, &dev_pitch, ncol * sizeof(float), nrow);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matC, &dev_pitch, ncol * sizeof(float), nrow);
    CUDA_CHECK_ERROR();

    cudaMemcpy2D(dev_matA, dev_pitch, matA, host_pitch, ncol * sizeof(float), nrow, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy2D(dev_matB, dev_pitch, matB, host_pitch, ncol * sizeof(float), nrow, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(32, 32 , 1);
    dim3 dimGrid((ncol + (dimBlock.x - 1)) / dimBlock.x, (nrow + (dimBlock.y - 1)) / dimBlock.y);
    ELAPSED_TIME_BEGIN(0);
    kernelAdd<<<dimGrid, dimBlock>>>(dev_matC, dev_matA, dev_matB, dev_pitch, nrow, ncol);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(0);

    cudaMemcpy2D(matC, host_pitch, dev_matC, dev_pitch, ncol * sizeof(float), nrow, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    cudaFree(dev_matA);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matB);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matC);
    CUDA_CHECK_ERROR();

    float sumA = getSum(matA, nrow * ncol);
    float sumB = getSum(matB, nrow * ncol);
    float sumC = getSum(matC, nrow * ncol);
    float diff = fabsf(sumC - (sumA + sumB));
    printf("matrix size = nrow * ncol = %d * %d\n", nrow, ncol);
    printf("sumC = %f\n", sumC);
    printf("sumA = %f\n", sumA);
    printf("sumB = %f\n", sumB);
    printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / (nrow * ncol) =  %f\n", diff / (nrow * ncol));
    printMat("matC", matC, nrow, ncol);
    printMat("matA", matA, nrow, ncol);
    printMat("matB", matB, nrow, ncol);

    free(matA);
    free(matB);
    free(matC);

    return 0;
}