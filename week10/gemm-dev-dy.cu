#include "./common.cpp"

// input parameters
const float alpha = 0.5f;
const float beta = -100.0f;
unsigned matsize = 4096; // num rows and also num cols

__global__ void kernelGEMM(float* matDst, float* matA, float* matB, float* matC, unsigned int matsize, int pitchInElem, const float alpha, const float beta) {
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gy < matsize && gx < matsize)  {
        unsigned idx = gy * pitchInElem + gx;
        float sum = 0;
        for (unsigned int k=0; k<matsize; k++) {
            unsigned idxA = gy * pitchInElem + k;
            unsigned idxB = k * pitchInElem + gx;
            sum += matA[idxA] * matB[idxB];
        }
        matDst[idx] = alpha * sum + beta * matC[idx];
    }
}

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		matsize = procArg( argv[0], argv[1], 4 );
		break;
	default:
		printf("usage: %s [matsize]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
    
    float* matA = (float*)malloc(matsize * matsize * sizeof(float));
    float* matB = (float*)malloc(matsize * matsize * sizeof(float));
    float* matC = (float*)malloc(matsize * matsize * sizeof(float));
    float* matZ = (float*)malloc(matsize * matsize * sizeof(float));

    srand(0);
    setNormalizedRandomData(matA, matsize * matsize);
    setNormalizedRandomData(matB, matsize * matsize);
    setNormalizedRandomData(matC, matsize * matsize);

    float* dev_matA = nullptr;
    float* dev_matB = nullptr;
    float* dev_matC = nullptr;
    float* dev_matZ = nullptr;
    size_t dPitch = 0;

    ELAPSED_TIME_BEGIN(1);
    cudaMallocPitch(&dev_matA, &dPitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matB, &dPitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matC, &dPitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matZ, &dPitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();

    size_t hostPitch = matsize * sizeof(float);
    cudaMemcpy2D(dev_matA, dPitch, matA, hostPitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy2D(dev_matB, dPitch, matB, hostPitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy2D(dev_matC, dPitch, matC, hostPitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(32, 32);
    dim3 dimGrid(div_up(matsize, dimBlock.x), div_up(matsize, dimBlock.y));
    assert(dPitch % sizeof(float) == 0);
    int dPitchInElem = dPitch / sizeof(float);
    CUDA_PRINT_CONFIG_2D( matsize, matsize );
    ELAPSED_TIME_BEGIN(0);
    kernelGEMM<<<dimGrid, dimBlock>>>(dev_matZ, dev_matA, dev_matB, dev_matC, matsize, dPitchInElem, alpha, beta);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(0);
    CUDA_CHECK_ERROR();
    
    cudaMemcpy2D(matZ, hostPitch, dev_matZ, dPitch, matsize * sizeof(float), matsize, cudaMemcpyDeviceToHost);
    ELAPSED_TIME_END(1);
    CUDA_CHECK_ERROR();
    
    cudaFree(dev_matA);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matB);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matC);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matZ);
    CUDA_CHECK_ERROR();

    float sumA = getSum(matA, matsize * matsize);
    float sumB = getSum(matB, matsize * matsize);
    float sumC = getSum(matC, matsize * matsize);
    float sumZ = getSum(matZ, matsize * matsize);
    printf("matrix size = matsize * matsize = %d * %d\n", matsize, matsize);
    printf("sumA = %f\n", sumA);
    printf("sumB = %f\n", sumB);
    printf("sumC = %f\n", sumC);
    printf("sumZ = %f\n", sumZ);
    printMat("matZ", matZ, matsize, matsize);
    printMat("matA", matA, matsize, matsize);
    printMat("matB", matB, matsize, matsize);
    printMat("matC", matC, matsize, matsize);

    free(matA);
    free(matB);
    free(matC);
    free(matZ);

    return 0;
}