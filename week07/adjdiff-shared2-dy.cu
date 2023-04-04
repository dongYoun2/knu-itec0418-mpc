#include "./common.cpp"

// input parameters
unsigned num = 16 * 1024 * 1024; // num data
unsigned blocksize = 1024; // shared mem buf size

__global__ void calcAdjDiff(float* dst, float* src, unsigned int n) {
    extern __shared__ float sharedData[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tx = threadIdx.x;

    if(idx < n) {
        sharedData[tx] = src[idx];
        __syncthreads();

        if(tx > 0) {
            dst[idx] = sharedData[tx] - sharedData[tx-1];
        } else if (idx > 0) {
            dst[idx] = sharedData[tx] - src[idx-1];
        } else {
            dst[idx] = sharedData[tx] - 0.0F;
        }
    }
}

int main(const int argc, const char* argv[]) {
    // argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		num = procArg( argv[0], argv[1], 1 );
		break;
	case 3:
		num = procArg( argv[0], argv[1], 1 );
		blocksize = procArg( argv[0], argv[2], 32, 1024 );
		break;
	default:
		printf("usage: %s [num] [blocksize]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}

    float* vecSrc = (float*)malloc(num * sizeof(float));
    float* vecDst = (float*)malloc(num * sizeof(float));

    srand(0);
    setNormalizedRandomData(vecSrc, num);

    float* dev_vecSrc = nullptr;
    float* dev_vecDst = nullptr;

    ELAPSED_TIME_BEGIN(1);

    cudaMalloc(&dev_vecSrc, num * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_vecDst, num * sizeof(float));
    CUDA_CHECK_ERROR();

    cudaMemcpy(dev_vecSrc, vecSrc, num * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(blocksize);
    dim3 dimGrid(div_up(num, blocksize));
    ELAPSED_TIME_BEGIN(0);
    calcAdjDiff<<<dimGrid, dimBlock, blocksize * sizeof(float)>>>(dev_vecDst, dev_vecSrc, num);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(0);

    cudaMemcpy(vecDst, dev_vecDst, num * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    cudaFree(dev_vecSrc);
    CUDA_CHECK_ERROR();
    cudaFree(dev_vecDst);
    CUDA_CHECK_ERROR();

    // check the result
	float sumSrc = getSum( vecSrc, num );
	float sumDst = getSum( vecDst, num );
	printf("sumSrc = %f\n", sumSrc);
	printf("sumDst = %f\n", sumDst);
	printVec( "vecSrc", vecSrc, num );
	printVec( "vecDst", vecDst, num );

    free(vecSrc);
    free(vecDst);

    return 0;
}