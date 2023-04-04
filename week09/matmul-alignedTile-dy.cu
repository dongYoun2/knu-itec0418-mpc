#include "./common.cpp"

// ASSUMPTION 1: matrix size는 32의 배수
// ASSUMPTION 2 tile 크기는 무조건 32 * 32

const unsigned TILE_WIDTH = 32;
unsigned matsize = 1024; // num rows and also num cols

__global__ void kernelMatMul(float* C, float* A, float* B, unsigned int n, unsigned int pitchInElem) {
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

    // REMIND: ASSUMPTION 1 떄문에 사실 이 바깥쪽 if문 필요없긴 함!
    if (gy < n && gx < n) {
        __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
        __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];
        unsigned int ty = threadIdx.y;
        unsigned int tx = threadIdx.x;
        
        unsigned int ntiles = n / TILE_WIDTH;
        float sum = 0.0f;
        for (int t = 0; t < ntiles; t++) {
            unsigned int x_A = t * TILE_WIDTH + tx;
            unsigned int idxA = gy * pitchInElem + x_A;
            sharedA[ty][tx] = A[idxA];

            unsigned int y_B = t * TILE_WIDTH + ty;
            unsigned int idxB = y_B * pitchInElem + gx;
            sharedB[ty][tx] = B[idxB];

            __syncthreads();
            for (unsigned int k = 0; k < TILE_WIDTH; k++) {
                sum += sharedA[ty][k] * sharedB[k][tx];
            }
            __syncthreads();
        }

        unsigned int idxC = gy * pitchInElem + gx;
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
	if (matsize % 32 != 0) {
		printf("%s: only accepts multiples of 32\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
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
    unsigned int pitchInElem = dpitch / sizeof(float);
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

    float sumA = getSum( matA, matsize * matsize );
	float sumB = getSum( matB, matsize * matsize );
	float sumC = getSum( matC, matsize * matsize );
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