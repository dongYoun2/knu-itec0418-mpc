#include "./common.cpp"

unsigned matsize = 4000; // num rows and also num cols

__global__ void kernelMatCpy( float* C, const float* A, int matsize, size_t pitch_in_elem ) {
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gy < matsize && gx < matsize) {
        int idx = gy * pitch_in_elem + gx;
        C[idx] = A[idx];
    }
}

int main(const int argc, const char* argv[]) {
    switch (argc) {
    case 1:
        break;
    case 2:
        matsize = procArg<int>(argv[0], argv[1], 4);
        break;
    default:
        printf("usage: %s [matsize]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    unsigned int matElemCnt = matsize * matsize;
    unsigned int totalMatSizeInBytes = matElemCnt * sizeof(float);
    unsigned int widthInBytes = matsize * sizeof(float);
    unsigned int width = matsize;
    unsigned int height = matsize;

    float* matSrc = (float*)malloc(totalMatSizeInBytes);
    float* matDst = (float*)malloc(totalMatSizeInBytes);

    srand(0);
    setNormalizedRandomData(matSrc, matElemCnt);

    float* dev_matSrc = nullptr;
    float* dev_matDst = nullptr;
    size_t devPitch = 0;

    ELAPSED_TIME_BEGIN(1);
    cudaMallocPitch(&dev_matSrc, &devPitch, widthInBytes, height);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matDst, &devPitch, widthInBytes, height);
    CUDA_CHECK_ERROR();

    float hostPitch = matsize * sizeof(float);
    cudaMemcpy2D(dev_matSrc, devPitch, matSrc, hostPitch, widthInBytes, height, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    
    dim3 dimBlock(32, 32);
    dim3 dimGrid(div_up(width, dimBlock.x), div_up(height, dimBlock.y));
    // QUESTION: pitchInElem이 딱 안 나누어 떨어지면 어떡하지?
    // 나누어 떨어진다고 가정해야함. 따라서 assert 같은 거 필요!
    assert(devPitch % sizeof(float) == 0);
    unsigned int pitchInElem = devPitch / sizeof(float);
    ELAPSED_TIME_BEGIN(0);
    kernelMatCpy<<<dimGrid, dimBlock>>>(dev_matDst, dev_matSrc, matsize, pitchInElem);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(0);

    cudaMemcpy2D(matDst, hostPitch, dev_matDst, devPitch, widthInBytes, height, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    cudaFree(dev_matSrc);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matDst);
    CUDA_CHECK_ERROR();

    float sumSrc = getSum(matSrc, matElemCnt);
    float sumDst = getSum(matDst, matElemCnt);
    float diff = fabsf(sumSrc - sumDst);

    printf("matrix size: %d * %d\n", width, height);
    printf("sumSrc: %f\n", sumSrc);
    printf("sumDst: %f\n", sumDst);
    printf("diff(sumSrc, sumDst): %f\n", diff);
    printf("diff(sumSrc, sumDst) / SIZE: %f\n", diff / (width * height));
    printMat("Src", matSrc, width, height);
    printMat("Dst", matDst, width, height);

    free(matSrc);
    free(matDst);


    return 0;
}