#include "./common.cpp"

int nrow = 10000;
int ncol = 10000;

__global__ void kernelAdd(float* dst, float* src1, float* src2, int nrow, int ncol) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < ncol && y < nrow) {
        // 첨에 반대로 함..
        // unsigned int idx = x * ncol + y;
        unsigned int idx = y * ncol + x;
        dst[idx] = src1[idx] + src2[idx];
    }
}

int main(int argc, char *argv[]) {
    switch(argc) {
    case 1:
    // 인자로 아무 것도 안 줬을 때
        break;
    case 2:
    // 인자 하나만 줬을 때 (정방행렬 가정)
        nrow = ncol = procArg<int>(argv[0], argv[1]);
        break;
    case 3:
    // 인자 두 개 주면 nrow, ncol 이라고 가정
        nrow = procArg<int>(argv[0], argv[1]);
        ncol = procArg<int>(argv[0], argv[2]);
        break;
    default:
		printf("usage: %s [nrow] [ncol]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    float *matA = (float*)malloc(nrow*ncol*sizeof(float));
    float *matB = (float*)malloc(nrow*ncol*sizeof(float));
    float *matC = (float*)malloc(nrow*ncol*sizeof(float));

    srand(42);
    setNormalizedRandomData(matA, nrow*ncol);
    setNormalizedRandomData(matB, nrow*ncol);

    float *dev_matA = nullptr;
    float *dev_matB = nullptr;
    float *dev_matC = nullptr;

    cudaMalloc(&dev_matA, nrow*ncol*sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_matB, nrow*ncol*sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_matC, nrow*ncol*sizeof(float));
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(1);
    cudaMemcpy(dev_matA, matA, nrow*ncol*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(dev_matB, matB, nrow*ncol*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(32, 32);
    int gx = (int)(ncol + (dimBlock.x - 1) / dimBlock.x);
    int gy = (int)(nrow + (dimBlock.y - 1) / dimBlock.y);
    dim3 dimGrid(gx, gy);
    // dim3 dimBlock(32, 32, 1);
	// dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x, (nrow + dimBlock.y - 1) / dimBlock.y, 1);
    ELAPSED_TIME_BEGIN(0);
    kernelAdd<<<dimGrid, dimBlock>>>(dev_matC, dev_matA, dev_matB, nrow, ncol);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(0);
    CUDA_CHECK_ERROR();

    cudaMemcpy(matC, dev_matC, nrow*ncol*sizeof(float), cudaMemcpyDeviceToHost);
    ELAPSED_TIME_END(1);

    cudaFree(dev_matA);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matB);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matC);
    CUDA_CHECK_ERROR();

    float sumA = getSum( matA, nrow * ncol );
	float sumB = getSum( matB, nrow * ncol );
	float sumC = getSum( matC, nrow * ncol );
	float diff = fabsf( sumC - (sumA + sumB) );
	printf("matrix size = nrow * ncol = %d * %d\n", nrow, ncol);
	printf("sumC = %f\n", sumC);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / (nrow * ncol) =  %f\n", diff / (nrow * ncol));
	printMat( "matC", matC, nrow, ncol );
	printMat( "matA", matA, nrow, ncol );
	printMat( "matB", matB, nrow, ncol );

    free(matA);
    free(matB);
    free(matC);

    return 0;
}