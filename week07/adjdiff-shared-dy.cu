#include "./common.cpp"

// input parameters
unsigned num = 16 * 1024 * 1024; // num data
const int MAX_THREAD_CNT = 1024;

__global__ void calcAdjDiff(float* dst, float* src, unsigned int n) {
    __shared__ float sharedMem[MAX_THREAD_CNT];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tx = threadIdx.x;
    if(i < n) {
        sharedMem[tx] = src[i];
        __syncthreads();
        
        if(tx > 0) {
            dst[i] = sharedMem[tx] - sharedMem[tx-1];
        } else if (i > 0 && tx == 0) {
            dst[i] = sharedMem[tx] - src[i-1];
        } else { // i == 0 (global index 가 0)
            dst[i] = sharedMem[tx] - 0.0F;
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
	default:
		printf("usage: %s [num]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}

    float* srcVec = (float*)malloc(num * sizeof(float));
    float* dstVec = (float*)malloc(num * sizeof(float));

    srand(0);
    setNormalizedRandomData(srcVec, num);
    
    float* dev_srcVec = nullptr;
    float* dev_dstVec = nullptr;

    ELAPSED_TIME_BEGIN(1);
    cudaMalloc(&dev_srcVec, num * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_dstVec, num * sizeof(float));
    CUDA_CHECK_ERROR();

    cudaMemcpy(dev_srcVec, srcVec, num * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(MAX_THREAD_CNT);
    dim3 dimGrid(div_up(num, dimBlock.x));
    ELAPSED_TIME_BEGIN(0);
    calcAdjDiff<<<dimGrid, dimBlock>>>(dev_dstVec, dev_srcVec, num);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(0);

    cudaMemcpy(dstVec, dev_dstVec, num * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    cudaFree(dev_srcVec);
    CUDA_CHECK_ERROR();
    cudaFree(dev_dstVec);
    CUDA_CHECK_ERROR();

    // check the result
	float sumA = getSum( srcVec, num );
	float sumB = getSum( dstVec, num );
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printVec( "srcVec", srcVec, num );
	printVec( "dstVec", dstVec, num );

    free(srcVec);
    free(dstVec);

    return 0;
}