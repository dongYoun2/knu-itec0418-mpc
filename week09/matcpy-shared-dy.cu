#include "./common.cpp"

unsigned matsize = 4000; // num rows and also num cols

__global__ void kernelMatCpy(float* dst, float* src, unsigned int n, size_t pitchInElem) {
    __shared__ float sdata[32][32];
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gy < n && gx < n) {
        unsigned int ty = threadIdx.y;
        unsigned int tx = threadIdx.x;
        unsigned int idx = gy * pitchInElem + gx;
        sdata[ty][tx] = src[idx];
        // REMIND: matrix copy 에 shared Mem. 사용할 때 __syncthreads()
        // 굳이 사용 안 해도 됨! (matrix copy 시 특정 thread에서
        // 다른 thread가 shared Mem.으로 옮겨온 데이터를 갖다 쓸 일이 없기 떄문!)
        // __syncthreads();
        dst[idx] = sdata[ty][tx];
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

    float* matSrc = (float*)malloc(matsize * matsize * sizeof(float));
    float* matDst = (float*)malloc(matsize * matsize * sizeof(float));

    srand(0);
    setNormalizedRandomData(matSrc, matsize * matsize);

    float* dev_matSrc = nullptr;
    float* dev_matDst = nullptr;
    size_t dpitch = 0;

    ELAPSED_TIME_BEGIN(1);
    cudaMallocPitch(&dev_matSrc, &dpitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matDst, &dpitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();

    size_t hostPitch = matsize * sizeof(float);
    cudaMemcpy2D(dev_matSrc, dpitch, matSrc, hostPitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(32, 32);
    dim3 dimGrid(div_up(matsize, dimBlock.x), div_up(matsize, dimBlock.y));
    assert(dpitch % sizeof(float) == 0);
    size_t pitchInElem = dpitch / sizeof(float);
    ELAPSED_TIME_BEGIN(0);
    kernelMatCpy<<<dimGrid, dimBlock>>>(dev_matDst, dev_matSrc, matsize, pitchInElem);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(0);

    cudaMemcpy2D(matDst, hostPitch, dev_matDst, dpitch, matsize * sizeof(float), matsize, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    cudaFree(dev_matSrc);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matDst);
    CUDA_CHECK_ERROR();

    float sumSrc = getSum(matSrc, matsize * matsize);
    float sumDst = getSum(matDst, matsize * matsize);
    float diff = fabsf(sumSrc - sumDst);

    printf("matrix size: %d * %d\n", matsize, matsize);
    printf("sumSrc: %f\n", sumSrc);
    printf("sumDst: %f\n", sumDst);
    printf("diff(sumSrc, sumDst): %f\n", diff);
    printf("diff(sumSrc, sumDst) / SIZE: %f\n", diff / (matsize * matsize));
    printMat("Src", matSrc, matsize, matsize);
    printMat("Dst", matDst, matsize, matsize);

    free(matSrc);
    free(matDst);

    return 0;
}
