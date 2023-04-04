#include "./common.cpp"

unsigned matsize = 1024; // num rows and also num cols

// REMIND: global Mem. 만 사용하는 이 코드는 임의의 matrix size 가능함!
__global__ void kernelMatMul( float* C, const float* A, const float* B, unsigned matsize, size_t pitch_in_elem ) {
    register unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    register unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gy < matsize && gx < matsize) {
        register unsigned int idxC = gy * pitch_in_elem + gx;
        register float sum = 0;
        for (register int k = 0; k < matsize; k++) {
            register unsigned int idxA = gy * pitch_in_elem + k;
            register unsigned int idxB = k * pitch_in_elem + gx;
            sum += A[idxA] * B[idxB];
        }
        C[idxC] = sum;
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

    srand(0);
    setNormalizedRandomData(matA, matsize * matsize);
    setNormalizedRandomData(matB, matsize * matsize);

    float* dev_matA = nullptr;
    float* dev_matB = nullptr;
    float* dev_matC = nullptr;
    size_t dpitch = 0;

    ELAPSED_TIME_BEGIN(1);
    cudaMallocPitch(&dev_matA, &dpitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matB, &dpitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matC, &dpitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();

    size_t hostPitch = matsize * sizeof(float);
    cudaMemcpy2D(dev_matA, dpitch, matA, hostPitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy2D(dev_matB, dpitch, matB, hostPitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(32, 32);
    dim3 dimGrid(div_up(matsize, dimBlock.x), div_up(matsize, dimBlock.y));
    assert(dpitch % sizeof(float) == 0);
    register unsigned int pitchInElem = dpitch / sizeof(float);
    ELAPSED_TIME_BEGIN(0);
    kernelMatMul<<<dimGrid, dimBlock>>>(dev_matC, dev_matA, dev_matB, matsize, pitchInElem);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(0);
    CUDA_CHECK_ERROR();

    cudaMemcpy2D(matC, hostPitch, dev_matC, dpitch, matsize * sizeof(float), matsize, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    cudaFree(dev_matA);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matB);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matC);
    CUDA_CHECK_ERROR();

    register float sumA = getSum( matA, matsize * matsize );
	register float sumB = getSum( matB, matsize * matsize );
	register float sumC = getSum( matC, matsize * matsize );
	printf("matrix size = matsize * matsize = %d * %d\n", matsize, matsize);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printMat( "matC", matC, matsize, matsize );
	printMat( "matA", matA, matsize, matsize );
	printMat( "matB", matB, matsize, matsize );

    free(matA);
	free(matB);
	free(matC);
    
    return 0;

}