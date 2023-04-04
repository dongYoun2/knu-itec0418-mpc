#include "./common.cpp"

// 임의의 matrix size와 임의의 tile 크기 (32, 16, 8, 20, ..)로 할 수 있도록!
// 심지어 24e-matmul-aligned2.cu는 tile 크기 변경할 수 있게 했지만 무조건 32이하인 2의 제곱수만 가능했는데
// 이 코드는 32이하의 아무 크기 다 가능함!!

const unsigned int MAX_TILE_WIDTH = 32;
unsigned TILE_WIDTH = 32;
unsigned matsize = 1024; // num rows and also num cols

__global__ void kernelMatMul(float* C, float* A, float* B, unsigned int n, unsigned int pitchInElem, unsigned int TILE_WIDTH) {
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

    // REMIND
    // 앞의 다른 matrix operartion 처럼( copy, transpose, ..)
    // if (gy < n && gx < n) 으로 전체를 걸어버리면 안 됨!!
    // 특정 스레드 블락이 shared Mem.(sharedA, sharedB) 에 A, B 올릴 때
    // 해당 스레드 블락내의 특정 스레드는 다음의 네가지 경우가 있을 수 있기 때문
    // 1. gy, gx 둘 다 범위 안 벗어남 -> sharedA: O, sharedB: O
    // 2. gy만 벗어남 -> sharedA: X, sharedB: O
    // 3. gx만 벗어남 -> sharedA: O, sharedB: X
    // 4. gy, gx 둘 다 범위 벗어남 -> sharedA: X, sharedB: X
    __shared__ float sharedA[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ float sharedB[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    unsigned int ty = threadIdx.y;
    unsigned int tx = threadIdx.x;

    unsigned int ntiles = (n + (TILE_WIDTH -1)) / TILE_WIDTH;
    float sum = 0.0f;
    for (int t = 0, remaining = n; t < ntiles; t++, remaining -= TILE_WIDTH) {
        // REMIND
        // 아래 주석 처리된 코드는 딱 나누어 떨어지는 경우와
        // 안 나누어 떨어지는 경우 모두를 대응하지 못함.
        // unsigned int nelem = (t == ntiles - 1 ? n % TILE_WIDTH : TILE_WIDTH);
        unsigned int nelem = min(remaining, TILE_WIDTH);
        if (gy < n && tx < nelem) {
            unsigned int x_A = t * TILE_WIDTH + tx;
            unsigned int idxA = gy * pitchInElem + x_A;
            sharedA[ty][tx] = A[idxA];
        }
        if (gx < n && ty < nelem) {
            unsigned int y_B = t * TILE_WIDTH + ty;
            unsigned int idxB = y_B * pitchInElem + gx;
            sharedB[ty][tx] = B[idxB];
        }
        __syncthreads();

        for (unsigned int k = 0; k < nelem; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();
    }

    unsigned int idxC = gy * pitchInElem + gx;
    C[idxC] = sum;
}

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		matsize = procArg( argv[0], argv[1], 4 );
		break;
    case 3:
        matsize = procArg(argv[0], argv[1], 4);
        TILE_WIDTH = procArg<int>(argv[0], argv[2], 1, MAX_TILE_WIDTH);
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
    unsigned int pitchInElem = dpitch / sizeof(float);
    ELAPSED_TIME_BEGIN(0);
    kernelMatMul<<<dimGrid, dimBlock>>>(dev_matC, dev_matA, dev_matB, matsize, pitchInElem, TILE_WIDTH);
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